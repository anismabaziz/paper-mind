"""
The ``VectorStore`` abstraction every vector backend implements.

The store API is intentionally small — upsert, query, delete — so swapping
Qdrant for another backend means implementing one class, not touching
retrieval code. ``query`` accepts an optional sparse vector for hybrid
dense+BM25 retrieval; backends without sparse support raise ``TypeError``
so the caller can degrade explicitly.
"""

from abc import ABC, abstractmethod


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
