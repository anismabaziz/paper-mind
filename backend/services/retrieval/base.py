"""
The ``VectorStore`` abstraction every vector backend implements.

The store API is intentionally small — upsert, query, delete — so swapping
Qdrant for another backend means implementing one class, not touching
retrieval code. ``query`` accepts an optional sparse vector for hybrid
dense+BM25 retrieval; backends without sparse support raise ``TypeError``
so the caller can degrade explicitly.
"""

from abc import ABC, abstractmethod


class VectorDimensionError(ValueError):
    """Typed error for embedding dimension mismatch — never auto-heals by wiping."""

    def __init__(self, expected: int | None = None, got: int | None = None, message: str | None = None):
        """Initialize with expected/got sizes and a remediation hint."""
        if message is None:
            if expected is not None and got is not None:
                message = (
                    f"Embedding dimension mismatch: collection expects {expected} but got {got}. "
                    "Delete embeddings via POST /delete-embeddings and re-ingest your Documents."
                )
            else:
                message = (
                    "Embedding dimension mismatch: the vector size does not match the collection. "
                    "Delete embeddings via POST /delete-embeddings and re-ingest your Documents."
                )
        super().__init__(message)
        self.expected = expected
        self.got = got


class VectorStore(ABC):
    """Dense (and optionally sparse-hybrid) vector index."""

    @abstractmethod
    def upsert(self, vectors) -> dict:
        """
        Store vectors with their payloads.

        ``vectors``: list of ``{"id": str, "values": list[float],
        "metadata": dict, "sparse_vector": {"indices": [], "values": []}}``.
        """

    @abstractmethod
    def query(self, vector, top_k, include_metadata=True, filter=None, **kwargs):
        """
        Return ``{"matches": [...]}`` with id/score/metadata entries.

        ``kwargs`` may carry ``sparse_vector`` for a hybrid dense+BM25 query;
        a backend that cannot execute it raises ``TypeError``.
        """

    @abstractmethod
    def delete(self, filter=None, delete_all=False) -> dict:
        """Delete by filter (or everything when ``delete_all`` is set)."""
