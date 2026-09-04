"""
Qdrant backend adapter with Pinecone-compatible shape.

Provides ``upsert``/``query``/``delete`` so ``VectorService`` and
``evaluation/evaluator.py`` can switch backends without code changes.

Collection ``pdf-index`` (``config.INDEX_NAME``) is created lazily with:

* dense vector ``size 1024`` ``distance Cosine``
* ``sparse_vectors`` with ``modifier IDF``
"""

import uuid

_QDRANT_DENSE_SIZE = 1024


class QdrantIndexAdapter:
    """
    Thin wrapper around a ``qdrant_client.QdrantClient`` that speaks the
    subset of the Pinecone ``Index`` API used in this repo.
    """

    def __init__(self, client, collection_name: str):
        self._client = client
        self._collection = collection_name
        self._ensured = False

    # ------------------------------------------------------------------ internals

    def _ensure_collection(self):
        if self._ensured:
            return
        try:
            exists = self._client.collection_exists(self._collection)
        except Exception:
            exists = False
        if exists:
            self._ensured = True
            return

        from qdrant_client.models import Distance, SparseIndexParams, SparseVectorParams, VectorParams

        try:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=_QDRANT_DENSE_SIZE, distance=Distance.COSINE),
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
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        must = []
        for key, value in filter_dict.items():
            must.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return Filter(must=must)

    # ------------------------------------------------------------------ Pinecone-compatible API

    def upsert(self, vectors):
        """
        ``vectors``: list of ``{"id": str, "values": list[float], "metadata": dict}``
        """
        if not vectors:
            return {"upserted": 0}
        self._ensure_collection()
        from qdrant_client.models import PointStruct

        points = []
        for v in vectors:
            vid = v.get("id") or str(uuid.uuid4())
            # Qdrant accepts UUID strings or ints; ensure valid UUID
            try:
                uuid.UUID(vid)
            except ValueError:
                vid = str(uuid.uuid5(uuid.NAMESPACE_URL, vid))
            values = v.get("values") or []
            payload = dict(v.get("metadata") or {})
            # Store size-agnostic: Qdrant will validate against collection size.
            # If caller sends 768-d Gemini vectors to a 1024-d collection, try
            # to upsert anyway; Qdrant will error which surfaces as a real
            # mismatch – callers on the free path use 1024-d local embeddings.
            points.append(PointStruct(id=vid, vector=values, payload=payload))

        # Handle size mismatch for backwards compat: if collection was created
        # with 1024 but vectors are different size, recreate with correct size.
        try:
            self._client.upsert(collection_name=self._collection, points=points, wait=True)
        except Exception as exc:
            msg = str(exc).lower()
            if "vector size" in msg or "dimension" in msg or "wrong vector size" in msg:
                # Recreate with the incoming size; preserve sparse config
                from qdrant_client.models import Distance, SparseIndexParams, SparseVectorParams, VectorParams

                try:
                    self._client.delete_collection(collection_name=self._collection)
                except Exception:
                    pass
                inferred = len(points[0].vector) if points and hasattr(points[0].vector, "__len__") else _QDRANT_DENSE_SIZE
                if not isinstance(inferred, int) or inferred <= 0:
                    inferred = _QDRANT_DENSE_SIZE
                self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(size=inferred, distance=Distance.COSINE),
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(
                            index=SparseIndexParams(on_disk=False),
                            modifier="idf",
                        )
                    },
                )
                self._ensured = True
                self._client.upsert(collection_name=self._collection, points=points, wait=True)
            else:
                raise
        return {"upserted": len(points)}

    def query(self, vector, top_k, include_metadata=True, filter=None):  # noqa: A002
        self._ensure_collection()
        q_filter = self._to_filter(filter)

        # Qdrant Python client: query_points is preferred in recent versions
        # Fall back to search for older clients
        try:
            # query_points (qdrant-client >=1.10)
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
                return {"matches": []}

        matches = []
        for p in points or []:
            # ScoredPoint has .payload, .score, .id
            if isinstance(p, dict):
                payload = p.get("payload") or p.get("metadata") or {}
                score = p.get("score", 0.0)
                pid = p.get("id")
            else:
                payload = getattr(p, "payload", {}) or {}
                score = getattr(p, "score", 0.0) or 0.0
                pid = getattr(p, "id", None)
            matches.append({"id": str(pid) if pid is not None else "", "score": float(score), "metadata": dict(payload)})

        return {"matches": matches}

    def delete(self, filter=None, delete_all=False):  # noqa: A002
        self._ensure_collection()
        if delete_all:
            # Fast path: delete collection and recreate empty one
            try:
                self._client.delete_collection(collection_name=self._collection)
            except Exception:
                pass
            self._ensured = False
            self._ensure_collection()
            return {"deleted": "all"}

        if filter is None:
            return {"deleted": 0}

        # Handle delete_all passed via dict: Pinecone allows
        # ``delete(delete_all=True)`` – our caller uses keyword.
        q_filter = self._to_filter(filter)
        if q_filter is None:
            return {"deleted": 0}
        try:
            self._client.delete(collection_name=self._collection, points_selector=q_filter, wait=True)
        except Exception:
            # Fallback: try scroll + delete by ids if filter delete not supported
            try:
                from qdrant_client.models import Filter as QFilter  # noqa: F401

                # Try delete with filter keyword variant
                self._client.delete(collection_name=self._collection, filter=q_filter, wait=True)
            except Exception:
                pass
        return {"deleted": "filtered"}
