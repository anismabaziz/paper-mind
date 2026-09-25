"""Qdrant adapter for durable dense and hashed term-frequency retrieval."""

from __future__ import annotations

import math
import threading
import uuid
from typing import Any, NoReturn, cast

from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    IsEmptyCondition,
    MatchValue,
    PayloadField,
    Modifier,
    PointStruct,
    Prefetch,
    Rrf,
    RrfQuery,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from services.retrieval.base import (
    RetrievalMethod,
    VectorDimensionError,
    VectorStore,
    VectorStoreConfigurationError,
    VectorStoreError,
    VectorStoreUnavailableError,
)
from services.retrieval.hybrid import RRF_K

DENSE_SIZE = 1024
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# Payload fields a stored Passage must carry for retrieval and citations.
_REPORTED_PAYLOAD_KEYS = (
    "content",
    "pdf_name",
    "chunk_index",
    "content_hash",
    "page_no",
    "sparse_method",
    "sparse_tokenizer_version",
    "index_generation",
)


def _is_empty_value(value: Any) -> bool:
    """Return whether a payload value carries no usable information."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set)):
        return not value
    return False


class QdrantIndexAdapter(VectorStore):
    """Store and retrieve named dense and sparse vectors in Qdrant."""

    def __init__(self, client, collection_name: str):
        """Bind the adapter to one Qdrant client and collection."""
        self._client = client
        self._collection = collection_name
        self._ensured = False
        self._lock = threading.RLock()

    @staticmethod
    def _raise_operation_error(
        exc: Exception,
        *,
        operation: str,
        vector: Any = None,
    ) -> NoReturn:
        if isinstance(exc, VectorStoreError):
            raise exc
        message = str(exc).lower()
        if "dimension" in message or "vector size" in message:
            got = len(vector) if hasattr(vector, "__len__") else None
            raise VectorDimensionError(expected=DENSE_SIZE, got=got) from exc
        if any(
            marker in message
            for marker in (
                "401",
                "403",
                "unauthorized",
                "forbidden",
                "api key",
                "credential",
            )
        ):
            raise VectorStoreConfigurationError(
                "Qdrant authentication or authorization failed"
            ) from exc
        if operation == "create_collection" and any(
            marker in message
            for marker in ("invalid", "schema", "vector name", "modifier")
        ):
            raise VectorStoreConfigurationError(
                f"Collection {exc!s} has an invalid vector schema"
            ) from exc
        raise VectorStoreUnavailableError(
            f"Vector store is unavailable during {operation}"
        ) from exc

    @staticmethod
    def _enum_value(value: Any) -> str:
        return str(getattr(value, "value", value)).lower()

    def _validate_existing_collection(self) -> None:
        try:
            collection = self._client.get_collection(self._collection)
        except Exception as exc:
            self._raise_operation_error(exc, operation="get_collection")
        params = getattr(getattr(collection, "config", None), "params", None)
        vectors = getattr(params, "vectors", None)
        sparse_vectors = getattr(params, "sparse_vectors", None)
        if not isinstance(vectors, dict) or not isinstance(sparse_vectors, dict):
            raise VectorStoreConfigurationError(
                f"Collection {self._collection!r} has an invalid vector schema"
            )
        dense = vectors.get(DENSE_VECTOR_NAME)
        sparse = sparse_vectors.get(SPARSE_VECTOR_NAME)
        if (
            dense is None
            or getattr(dense, "size", None) != DENSE_SIZE
            or self._enum_value(getattr(dense, "distance", None))
            != self._enum_value(Distance.COSINE)
        ):
            raise VectorStoreConfigurationError(
                f"Collection {self._collection!r} has an incompatible dense vector schema"
            )
        if (
            sparse is None
            or self._enum_value(getattr(sparse, "modifier", None)) != "idf"
        ):
            raise VectorStoreConfigurationError(
                f"Collection {self._collection!r} has an incompatible sparse vector schema"
            )

    def _ensure_collection(self, *, create: bool = True) -> None:
        if self._ensured:
            return
        with self._lock:
            if self._ensured:
                return
            try:
                exists = self._client.collection_exists(self._collection)
            except Exception as exc:
                self._raise_operation_error(exc, operation="collection_exists")
            if exists:
                self._validate_existing_collection()
                self._ensured = True
                return
            if not create:
                raise VectorStoreConfigurationError(
                    f"Collection {self._collection!r} does not exist"
                )
            try:
                self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config={
                        DENSE_VECTOR_NAME: VectorParams(
                            size=DENSE_SIZE,
                            distance=Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        SPARSE_VECTOR_NAME: SparseVectorParams(
                            index=SparseIndexParams(on_disk=False),
                            modifier=Modifier.IDF,
                        )
                    },
                )
            except Exception as exc:
                message = str(exc).lower()
                if "already exists" in message or "exists" in message:
                    self._validate_existing_collection()
                    self._ensured = True
                    return
                self._raise_operation_error(exc, operation="create_collection")
            self._ensured = True

    @staticmethod
    def _to_filter(filter_dict: dict | None) -> Filter | None:
        if not filter_dict:
            return None
        return Filter(
            must=[
                (
                    IsEmptyCondition(is_empty=PayloadField(key=key))
                    if value is None
                    else FieldCondition(key=key, match=MatchValue(value=value))
                )
                for key, value in filter_dict.items()
            ]
        )

    @staticmethod
    def _normalize_sparse(sparse: Any) -> dict[str, list]:
        if hasattr(sparse, "indices") and hasattr(sparse, "values"):
            indices = list(sparse.indices)
            values = list(sparse.values)
        elif isinstance(sparse, dict):
            indices = list(sparse.get("indices", []))
            values = list(sparse.get("values", []))
        else:
            raise VectorStoreConfigurationError("Sparse query vector is missing")
        if len(indices) != len(values):
            raise VectorStoreConfigurationError(
                "Sparse vector indices and values must have equal length"
            )
        if any(
            not isinstance(index, int) or isinstance(index, bool) for index in indices
        ):
            raise VectorStoreConfigurationError(
                "Sparse vector indices must be integers"
            )
        if any(index < 0 for index in indices):
            raise VectorStoreConfigurationError(
                "Sparse vector indices must be non-negative"
            )
        if indices != sorted(set(indices)):
            raise VectorStoreConfigurationError(
                "Sparse vector indices must be sorted and unique"
            )
        if any(not isinstance(value, (int, float)) for value in values):
            raise VectorStoreConfigurationError("Sparse vector values must be numbers")
        finite_values = [float(value) for value in values]
        if not all(math.isfinite(value) for value in finite_values):
            raise VectorStoreConfigurationError("Sparse vector values must be finite")
        return {"indices": indices, "values": finite_values}

    @staticmethod
    def _to_matches(points) -> list[dict]:
        matches = []
        for point in points or []:
            if isinstance(point, dict):
                payload = point.get("payload") or point.get("metadata") or {}
                score = point.get("score", 0.0)
                point_id = point.get("id")
            else:
                payload = getattr(point, "payload", {}) or {}
                score = getattr(point, "score", 0.0) or 0.0
                point_id = getattr(point, "id", None)
            matches.append(
                {
                    "id": str(point_id) if point_id is not None else "",
                    "score": float(score),
                    "metadata": dict(payload),
                }
            )
        return matches

    def upsert(self, vectors) -> dict:
        """Store dense and sparse representations for each vector record."""
        if not vectors:
            return {"upserted": 0}
        self._ensure_collection()
        points = []
        for vector in vectors:
            point_id = vector.get("id") or str(uuid.uuid4())
            try:
                uuid.UUID(point_id)
            except (AttributeError, TypeError, ValueError):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(point_id)))
            dense = vector.get("values")
            if not isinstance(dense, (list, tuple)):
                raise VectorStoreConfigurationError("Dense vector values are missing")
            if len(dense) != DENSE_SIZE:
                raise VectorDimensionError(expected=DENSE_SIZE, got=len(dense))
            sparse = self._normalize_sparse(
                vector.get("sparse_vector") or vector.get("sparse")
            )
            sparse_query = SparseVector(
                indices=sparse["indices"],
                values=sparse["values"],
            )
            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        DENSE_VECTOR_NAME: list(dense),
                        SPARSE_VECTOR_NAME: sparse_query,
                    },
                    payload=dict(vector.get("metadata") or {}),
                )
            )
        try:
            self._client.upsert(
                collection_name=self._collection,
                points=points,
                wait=True,
            )
        except Exception as exc:
            self._raise_operation_error(exc, operation="upsert", vector=dense)
        return {"upserted": len(vectors)}

    def _query_points(
        self,
        vector,
        sparse: dict[str, list],
        method: RetrievalMethod,
        top_k: int,
        q_filter: Filter | None,
    ) -> list[dict]:
        try:
            if method == "hybrid":
                sparse_query = SparseVector(
                    indices=sparse["indices"],
                    values=sparse["values"],
                )
                result = self._client.query_points(
                    collection_name=self._collection,
                    prefetch=[
                        Prefetch(
                            query=vector,
                            using=DENSE_VECTOR_NAME,
                            limit=top_k,
                        ),
                        Prefetch(
                            query=sparse_query,
                            using=SPARSE_VECTOR_NAME,
                            limit=top_k,
                        ),
                    ],
                    query=RrfQuery(rrf=Rrf(k=RRF_K)),
                    limit=top_k,
                    query_filter=q_filter,
                    with_payload=True,
                )
            elif method == "sparse":
                result = self._client.query_points(
                    collection_name=self._collection,
                    query=SparseVector(
                        indices=sparse["indices"],
                        values=sparse["values"],
                    ),
                    using=SPARSE_VECTOR_NAME,
                    limit=top_k,
                    query_filter=q_filter,
                    with_payload=True,
                )
            else:
                result = self._client.query_points(
                    collection_name=self._collection,
                    query=vector,
                    using=DENSE_VECTOR_NAME,
                    limit=top_k,
                    query_filter=q_filter,
                    with_payload=True,
                )
        except Exception as exc:
            self._raise_operation_error(exc, operation="query", vector=vector)
        return self._to_matches(getattr(result, "points", result))

    def query(
        self,
        vector,
        top_k,
        include_metadata=True,
        filter=None,
        **kwargs,
    ) -> dict:
        """Run the selected retrieval method and report its outcome."""
        self._ensure_collection(create=False)
        sparse_value = kwargs.get("sparse_vector") or kwargs.get("sparse")
        sparse = (
            self._normalize_sparse(sparse_value) if sparse_value is not None else None
        )
        method_value = kwargs.get("method") or ("hybrid" if sparse else "dense")
        if method_value not in {"dense", "sparse", "hybrid"}:
            raise VectorStoreConfigurationError(
                f"Unsupported retrieval method: {method_value}"
            )
        method = cast(RetrievalMethod, method_value)
        if method != "dense" and (sparse is None or not sparse["indices"]):
            raise VectorStoreConfigurationError(
                f"{method} retrieval requires a non-empty sparse query vector"
            )
        matches = self._query_points(
            vector,
            sparse or {"indices": [], "values": []},
            method,
            top_k,
            self._to_filter(filter),
        )
        return {
            "matches": matches,
            "method": method,
            "outcome": "success" if matches else "empty",
        }

    @staticmethod
    def _report_point(point: Any, value_keys: tuple[str, ...]) -> dict[str, Any]:
        """Describe one stored point for a generation report."""
        if isinstance(point, dict):
            payload = point.get("payload") or {}
            vectors = point.get("vector", point.get("vectors"))
        else:
            payload = getattr(point, "payload", None) or {}
            vectors = getattr(point, "vector", None)
            if vectors is None:
                vectors = getattr(point, "vectors", None)
        dense = vectors.get(DENSE_VECTOR_NAME) if isinstance(vectors, dict) else None
        sparse = vectors.get(SPARSE_VECTOR_NAME) if isinstance(vectors, dict) else None
        sparse_indices = getattr(sparse, "indices", None)
        if sparse_indices is None and isinstance(sparse, dict):
            sparse_indices = sparse.get("indices")
        pages: set[int] = set()
        page_no = payload.get("page_no")
        if isinstance(page_no, int) and not isinstance(page_no, bool):
            pages.add(page_no)
        return {
            "has_dense": bool(dense),
            "has_sparse": bool(sparse_indices),
            "payload": payload,
            "pages": pages,
            "value_keys": {
                key: payload.get(key) for key in value_keys if key in payload
            },
        }

    def generation_report(
        self,
        filter: dict | None = None,
        limit: int = 1000,
        value_keys: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """
        Describe what one index generation actually stores.

        The report is an inventory, not a verdict: counting, vector presence,
        payload completeness, and Page provenance are aggregated here, while
        the policy that decides whether the generation may be activated lives
        in :class:`services.retrieval.vector_service.VectorService`.
        """
        self._ensure_collection(create=False)
        total = self.count(filter=filter)
        try:
            scrolled = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=self._to_filter(filter),
                limit=limit,
                with_payload=True,
                with_vectors=True,
            )
        except Exception as exc:
            self._raise_operation_error(exc, operation="scroll")
        points = scrolled[0] if isinstance(scrolled, tuple) else scrolled
        points = list(points or [])
        without_dense = 0
        without_sparse = 0
        missing_payload: dict[str, int] = {}
        empty_payload: dict[str, int] = {}
        pages: set[int] = set()
        distinct_values: dict[str, set[Any]] = {key: set() for key in value_keys}
        for point in points:
            described = self._report_point(point, value_keys)
            if not described["has_dense"]:
                without_dense += 1
            if not described["has_sparse"]:
                without_sparse += 1
            pages |= described["pages"]
            for key, value in described["value_keys"].items():
                distinct_values[key].add(value)
            payload = described["payload"]
            for key in _REPORTED_PAYLOAD_KEYS:
                if key not in payload:
                    missing_payload[key] = missing_payload.get(key, 0) + 1
                elif _is_empty_value(payload[key]):
                    empty_payload[key] = empty_payload.get(key, 0) + 1
        return {
            "total": total,
            "inspected": len(points),
            "truncated": len(points) < total,
            "without_dense": without_dense,
            "without_sparse": without_sparse,
            "missing_payload": missing_payload,
            "empty_payload": empty_payload,
            "pages": sorted(pages),
            "distinct_values": {
                key: sorted(value, key=str) for key, value in distinct_values.items()
            },
        }

    def count(self, filter=None) -> int:
        """Count points matching one payload filter."""
        self._ensure_collection(create=False)
        try:
            result = self._client.count(
                collection_name=self._collection,
                count_filter=self._to_filter(filter),
                exact=True,
            )
        except Exception as exc:
            self._raise_operation_error(exc, operation="count")
        return int(getattr(result, "count", result))

    def delete(self, filter=None, delete_all=False) -> dict:
        """Delete all points or the points matching one payload filter."""
        self._ensure_collection()
        if delete_all:
            try:
                self._client.delete_collection(collection_name=self._collection)
            except Exception as exc:
                self._raise_operation_error(exc, operation="delete_collection")
            with self._lock:
                self._ensured = False
            self._ensure_collection()
            return {"deleted": "all"}
        q_filter = self._to_filter(filter)
        if q_filter is None:
            return {"deleted": 0}
        try:
            self._client.delete(
                collection_name=self._collection,
                points_selector=q_filter,
                wait=True,
            )
        except Exception as exc:
            self._raise_operation_error(exc, operation="delete")
        return {"deleted": "filtered"}
