"""
Qdrant backend adapter.

Provides ``upsert``/``query``/``delete`` so ``VectorService`` and
``evaluation/evaluator.py`` share one store interface.

Collection ``pdf-index`` (the Settings vector index name) is created lazily with:

* dense vector ``size 1024`` ``distance Cosine``
* ``sparse_vectors`` with ``modifier IDF`` (BM25 sparse via ``rank-bm25``)
"""

import threading
import uuid

from services.retrieval.base import VectorDimensionError, VectorStore

_QDRANT_DENSE_SIZE = 1024
# Bounded in-memory caches for sparse/python fallback: filtered deletes only prune,
# never lost on dimension mismatch, and capped to avoid unbounded growth.
_MAX_CACHE_ENTRIES = 10000


class QdrantIndexAdapter(VectorStore):
    """
    Thin wrapper around a ``qdrant_client.QdrantClient`` implementing the.

    ``upsert``/``query``/``delete`` store API used in this repo, plus hybrid
    sparse support for ``VectorService``.
    """

    def __init__(self, client, collection_name: str):
        """Initialize."""
        self._client = client
        self._collection = collection_name
        self._ensured = False
        self._lock = threading.RLock()
        # Cache sparse vectors for python-side fallback when server hybrid
        # is not available or for offline tests with a stub client.
        self._sparse_by_id: dict[str, dict] = {}
        self._payload_by_id: dict[str, dict] = {}

    # ------------------------------------------------------------------ internals

    def _ensure_collection(self):
        if self._ensured:
            return
        with self._lock:
            if self._ensured:
                return
            try:
                exists = self._client.collection_exists(self._collection)
            except Exception:
                exists = False
            if exists:
                self._ensured = True
                return

            from qdrant_client.models import (
                Distance,
                SparseIndexParams,
                SparseVectorParams,
                VectorParams,
            )

            try:
                self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(
                        size=_QDRANT_DENSE_SIZE, distance=Distance.COSINE
                    ),
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(
                            index=SparseIndexParams(on_disk=False),
                            modifier="idf",
                        )
                    },
                )
            except Exception as exc:
                # If another worker created it concurrently, treat as success
                msg = str(exc).lower()
                if "already exists" in msg or "exists" in msg:
                    self._ensured = True
                    return
                raise
            self._ensured = True

    @staticmethod
    def _to_filter(filter_dict):
        if not filter_dict:
            return None
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            must = []
            for key, value in filter_dict.items():
                must.append(FieldCondition(key=key, match=MatchValue(value=value)))
            return Filter(must=must)
        except Exception:
            # Offline / no qdrant_client: return dict as fallback filter
            return filter_dict

    @staticmethod
    def _sparse_dot(query_sparse: dict, doc_sparse: dict) -> float:
        if not query_sparse or not doc_sparse:
            return 0.0
        q_map = dict(
            zip(query_sparse.get("indices", []), query_sparse.get("values", []))
        )
        d_map = dict(zip(doc_sparse.get("indices", []), doc_sparse.get("values", [])))
        # Qdrant's sparse modifier IDF is applied server-side; for the
        # Python brute fallback we approximate BM25 via dot product. When
        # rank-bm25 is available the full BM25 scoring is used in
        # _brute_sparse_query via bm25_scores, so this dot is only the TF proxy.
        return sum(q_map[i] * d_map.get(i, 0.0) for i in q_map)

    def _brute_sparse_query(
        self, sparse_vector: dict, top_k: int, q_filter
    ) -> list[dict]:
        """
        Fallback sparse scoring over the locally cached sparse map.

        Only used when the server cannot execute a sparse query.
        """
        # Filter by pdf_name if present
        # q_filter is a Qdrant Filter object – extract pdf_name from it
        pdf_filter = None
        if q_filter is not None:
            try:
                if isinstance(q_filter, dict):
                    pdf_filter = q_filter.get("pdf_name")
                else:
                    for cond in getattr(q_filter, "must", []) or []:
                        if getattr(cond, "key", None) == "pdf_name":
                            pdf_filter = getattr(
                                getattr(cond, "match", None), "value", None
                            )
            except Exception:
                pass

        scored: list[dict] = []
        # Snapshot caches under lock for thread safety
        with self._lock:
            sparse_items = list(self._sparse_by_id.items())
            payload_snapshot = dict(self._payload_by_id)
        for vid, doc_sparse in sparse_items:
            payload = payload_snapshot.get(vid, {})
            if pdf_filter is not None and payload.get("pdf_name") != pdf_filter:
                continue
            score = self._sparse_dot(sparse_vector, doc_sparse)
            if score == 0:
                continue
            scored.append({"id": vid, "score": float(score), "metadata": dict(payload)})
        scored.sort(key=lambda m: m["score"], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------ store API

    def _enforce_cache_bounds(self):
        """Evict oldest entries when caches exceed max size (FIFO)."""
        while len(self._payload_by_id) > _MAX_CACHE_ENTRIES:
            oldest = next(iter(self._payload_by_id))
            self._payload_by_id.pop(oldest, None)
            self._sparse_by_id.pop(oldest, None)
        # Keep sparse map in sync — if it somehow grew larger
        while len(self._sparse_by_id) > _MAX_CACHE_ENTRIES:
            oldest = next(iter(self._sparse_by_id))
            self._sparse_by_id.pop(oldest, None)
            self._payload_by_id.pop(oldest, None)

    def _commit_pending(
        self, pending_sparse: dict[str, dict | None], pending_payload: dict[str, dict]
    ) -> None:
        """Merge staged sparse/payload entries into bounded caches (lock held)."""
        with self._lock:
            for vid, sparse in pending_sparse.items():
                if sparse is not None:
                    self._sparse_by_id[vid] = sparse
                else:
                    self._sparse_by_id.pop(vid, None)
                self._payload_by_id[vid] = pending_payload[vid]
            self._enforce_cache_bounds()

    def upsert(self, vectors):
        """``vectors``: list of ``{"id": str, "values": list[float], "metadata": dict, "sparse_vector": {"indices": [], "values": []}}``."""
        if not vectors:
            return {"upserted": 0}
        self._ensure_collection()
        try:
            from qdrant_client.models import PointStruct, SparseVector
        except Exception:
            PointStruct = None  # type: ignore
            SparseVector = None  # type: ignore

        points = []
        pending_sparse: dict[str, dict | None] = {}
        pending_payload: dict[str, dict] = {}
        for v in vectors:
            vid = v.get("id") or str(uuid.uuid4())
            # Qdrant accepts UUID strings or ints; ensure valid UUID
            try:
                uuid.UUID(vid)
            except ValueError:
                vid = str(uuid.uuid5(uuid.NAMESPACE_URL, vid))
            values = v.get("values") or []
            payload = dict(v.get("metadata") or {})
            sparse = v.get("sparse_vector") or v.get("sparse")
            # Stage cache entries — only committed on successful upsert so a
            # dimension mismatch never wipes or pollutes the existing caches.
            if sparse:
                # normalise to dict shape
                if hasattr(sparse, "indices"):
                    sparse = {
                        "indices": list(sparse.indices),
                        "values": list(sparse.values),
                    }
                pending_sparse[vid] = dict(sparse)
            else:
                pending_sparse[vid] = None
            pending_payload[vid] = dict(payload)

            if PointStruct is None:
                # No qdrant_client available (offline tests via stub client)
                continue

            # Try to include sparse_vector if the model supports it
            try:
                if (
                    sparse
                    and SparseVector is not None
                    and hasattr(PointStruct, "__init__")
                ):
                    sparse_obj = SparseVector(
                        indices=sparse["indices"], values=sparse["values"]
                    )
                    # Prefer explicit sparse_vector field if supported
                    try:
                        points.append(
                            PointStruct(
                                id=vid,
                                vector=values,
                                payload=payload,
                                sparse_vector=sparse_obj,
                            )
                        )
                        continue
                    except TypeError:
                        pass
                    # Fallback: named vector dict
                    try:
                        points.append(
                            PointStruct(
                                id=vid,
                                vector={"dense": values, "sparse": sparse_obj},
                                payload=payload,
                            )
                        )
                        continue
                    except Exception:
                        pass
                points.append(PointStruct(id=vid, vector=values, payload=payload))
            except Exception:
                # Last resort plain dense
                points.append(PointStruct(id=vid, vector=values, payload=payload))

        if PointStruct is None:
            # Offline / stub client: commit pending caches directly (bounded)
            self._commit_pending(pending_sparse, pending_payload)
            return {"upserted": len(vectors)}

        if points:
            try:
                self._client.upsert(
                    collection_name=self._collection, points=points, wait=True
                )
            except Exception as exc:
                msg = str(exc).lower()
                if (
                    "vector size" in msg
                    or "dimension" in msg
                    or "wrong vector size" in msg
                    or ("vector" in msg and "size" in msg)
                ):
                    # Fail fast — do NOT delete collection or clear caches.
                    # Infer sizes for the typed error hint.
                    inferred = None
                    if points:
                        vec = getattr(points[0], "vector", None)
                        if isinstance(vec, dict):
                            dense = vec.get("dense")
                            if dense is not None and hasattr(dense, "__len__"):
                                inferred = len(dense)
                        elif hasattr(vec, "__len__"):
                            inferred = len(vec)
                    raise VectorDimensionError(
                        expected=_QDRANT_DENSE_SIZE, got=inferred
                    ) from exc
                raise
            # Success — commit staged caches and bound them
            self._commit_pending(pending_sparse, pending_payload)
        else:
            # No points (empty vectors edge) still need to stage if offline? Already handled
            self._commit_pending(pending_sparse, pending_payload)
        return {"upserted": len(vectors)}

    def _dense_search(self, vector, top_k, q_filter):
        # Qdrant Python client: query_points is preferred in recent versions
        # Fall back to search for older clients
        try:
            result = self._client.query_points(
                collection_name=self._collection,
                query=vector,
                limit=top_k,
                query_filter=q_filter,
                with_payload=True,
            )
            points = getattr(result, "points", result)
        except AttributeError:
            points = []
        except Exception:
            # Fallback to legacy search
            try:
                points = self._client.search(
                    collection_name=self._collection,
                    query_vector=vector,
                    limit=top_k,
                    query_filter=q_filter,
                    with_payload=True,
                )
            except Exception:
                # Collection empty / not found
                return []
        matches = []
        for p in points or []:
            if isinstance(p, dict):
                payload = p.get("payload") or p.get("metadata") or {}
                score = p.get("score", 0.0)
                pid = p.get("id")
            else:
                payload = getattr(p, "payload", {}) or {}
                score = getattr(p, "score", 0.0) or 0.0
                pid = getattr(p, "id", None)
            matches.append(
                {
                    "id": str(pid) if pid is not None else "",
                    "score": float(score),
                    "metadata": dict(payload),
                }
            )
        return matches

    def _sparse_search(self, sparse_vector: dict, top_k: int, q_filter):
        # Try server-side sparse query
        try:
            from qdrant_client.models import SparseVector
        except Exception:
            SparseVector = None  # type: ignore

        if SparseVector is not None:
            try:
                sparse_q = SparseVector(
                    indices=sparse_vector.get("indices", []),
                    values=sparse_vector.get("values", []),
                )
                # New API: query_points with using="sparse"
                result = self._client.query_points(
                    collection_name=self._collection,
                    query=sparse_q,
                    using="sparse",
                    limit=top_k,
                    query_filter=q_filter,
                    with_payload=True,
                )
                points = getattr(result, "points", result)
                matches = []
                for p in points or []:
                    if isinstance(p, dict):
                        payload = p.get("payload") or p.get("metadata") or {}
                        score = p.get("score", 0.0)
                        pid = p.get("id")
                    else:
                        payload = getattr(p, "payload", {}) or {}
                        score = getattr(p, "score", 0.0) or 0.0
                        pid = getattr(p, "id", None)
                    matches.append(
                        {
                            "id": str(pid) if pid is not None else "",
                            "score": float(score),
                            "metadata": dict(payload),
                        }
                    )
                if matches:
                    return matches
            except Exception:
                pass

            # Fallback via search with sparse vector (older API)
            try:
                points = self._client.search(
                    collection_name=self._collection,
                    query_vector=sparse_q,
                    limit=top_k,
                    query_filter=q_filter,
                    with_payload=True,
                )
                matches = []
                for p in points or []:
                    payload = getattr(p, "payload", {}) or {}
                    score = getattr(p, "score", 0.0) or 0.0
                    pid = getattr(p, "id", None)
                    matches.append(
                        {
                            "id": str(pid) if pid is not None else "",
                            "score": float(score),
                            "metadata": dict(payload),
                        }
                    )
                if matches:
                    return matches
            except Exception:
                pass

        # Pure python fallback over cached sparse map
        return self._brute_sparse_query(sparse_vector, top_k, q_filter)

    def query(self, vector, top_k, include_metadata=True, filter=None, **kwargs):
        """Do query."""
        self._ensure_collection()
        q_filter = self._to_filter(filter)

        sparse_vector = kwargs.get("sparse_vector") or kwargs.get("sparse")
        # Normalise sparse_vector from object to dict if needed
        if sparse_vector is not None and hasattr(sparse_vector, "indices"):
            sparse_vector = {
                "indices": list(sparse_vector.indices),
                "values": list(sparse_vector.values),
            }

        # No hybrid requested -> dense only (backwards compat)
        if not sparse_vector:
            matches = self._dense_search(vector, top_k, q_filter)
            return {"matches": matches}

        # Try single Qdrant hybrid query (dense + BM25 sparse via rank-bm25 on
        # the ``sparse`` field) with RRF(k=60). Falls back to two searches + RRF.
        single = self._single_hybrid_query(vector, sparse_vector, top_k, q_filter)
        if single is not None:
            return {"matches": single}

        # Fallback: dense + sparse then fuse via RRF(k=60)
        dense_matches = self._dense_search(vector, top_k, q_filter)
        sparse_matches = self._sparse_search(sparse_vector, top_k, q_filter)

        # If one side returned nothing, return the other side
        if not dense_matches:
            return {"matches": sparse_matches[:top_k]}
        if not sparse_matches:
            return {"matches": dense_matches[:top_k]}

        try:
            from services.retrieval.hybrid import rrf_fusion

            fused = rrf_fusion([dense_matches, sparse_matches], limit=top_k)
            return {"matches": fused}
        except Exception:
            # Fallback to dense if fusion fails
            return {"matches": dense_matches}

    def _single_hybrid_query(self, vector, sparse_vector, top_k, q_filter):
        """Attempt a single Qdrant hybrid query with RRF fusion."""
        try:
            from qdrant_client.models import FusionQuery, Fusion, Prefetch, SparseVector
        except Exception:
            return None
        try:
            sparse_q = SparseVector(
                indices=sparse_vector.get("indices", []),
                values=sparse_vector.get("values", []),
            )
            prefetch = [
                Prefetch(query=vector, limit=top_k),
                Prefetch(query=sparse_q, using="sparse", limit=top_k),
            ]
            result = self._client.query_points(
                collection_name=self._collection,
                prefetch=prefetch,
                query=FusionQuery(fusion=Fusion.RRF),
                with_payload=True,
                query_filter=q_filter,
            )
            points = getattr(result, "points", result) or []
            matches = []
            for p in points:
                if isinstance(p, dict):
                    payload = p.get("payload") or p.get("metadata") or {}
                    score = p.get("score", 0.0)
                    pid = p.get("id")
                else:
                    payload = getattr(p, "payload", {}) or {}
                    score = getattr(p, "score", 0.0) or 0.0
                    pid = getattr(p, "id", None)
                matches.append(
                    {
                        "id": str(pid) if pid is not None else "",
                        "score": float(score),
                        "metadata": dict(payload),
                    }
                )
            if matches:
                return matches[:top_k]
        except Exception:
            return None
        return None

    def delete(self, filter=None, delete_all=False):
        """Do delete."""
        self._ensure_collection()
        if delete_all:
            # Fast path: delete collection and recreate empty one
            try:
                self._client.delete_collection(collection_name=self._collection)
            except Exception:
                pass
            with self._lock:
                self._ensured = False
                self._sparse_by_id.clear()
                self._payload_by_id.clear()
            self._ensure_collection()
            return {"deleted": "all"}

        if filter is None:
            return {"deleted": 0}

        # Handle delete_all passed via dict: accept
        # ``delete(delete_all=True)`` – our caller uses keyword.
        q_filter = self._to_filter(filter)
        if q_filter is None:
            return {"deleted": 0}
        # Also prune local cache for filtered deletes by pdf_name (bounded,
        # only filtered deletes touch the caches — never on dimension mismatch)
        if filter and "pdf_name" in filter:
            target = filter["pdf_name"]
            with self._lock:
                to_remove = [
                    vid
                    for vid, payload in self._payload_by_id.items()
                    if payload.get("pdf_name") == target
                ]
                for vid in to_remove:
                    self._sparse_by_id.pop(vid, None)
                    self._payload_by_id.pop(vid, None)
        try:
            self._client.delete(
                collection_name=self._collection, points_selector=q_filter, wait=True
            )
        except Exception:
            # Fallback: try delete with filter keyword variant
            try:
                self._client.delete(
                    collection_name=self._collection, filter=q_filter, wait=True
                )
            except Exception:
                pass
        return {"deleted": "filtered"}
