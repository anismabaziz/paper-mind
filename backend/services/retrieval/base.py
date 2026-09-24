"""Vector-store contracts, retrieval outcomes, and typed failures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

RetrievalMethod = Literal["dense", "sparse", "hybrid"]
RetrievalOutcome = Literal["success", "empty"]


@dataclass(frozen=True)
class RetrievalResult:
    """Shaped evidence plus the method and outcome that produced it."""

    sources: list[dict]
    method: RetrievalMethod
    outcome: RetrievalOutcome


class VectorStoreError(RuntimeError):
    """Base error for vector-store operations."""


class VectorStoreUnavailableError(VectorStoreError):
    """The configured vector store cannot be reached."""


class VectorStoreConfigurationError(VectorStoreError):
    """The vector store or collection is configured incorrectly."""


class VectorDimensionError(VectorStoreError, ValueError):
    """Typed error for embedding dimension mismatch — never auto-heals by wiping."""

    def __init__(
        self,
        expected: int | None = None,
        got: int | None = None,
        message: str | None = None,
    ):
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
    """Dense, sparse, and hybrid vector index."""

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
        Return matches with the method and empty or success outcome.

        ``kwargs`` may carry ``sparse_vector`` and an explicit dense, sparse,
        or hybrid ``method``. Invalid requests and service failures raise typed
        errors rather than changing the selected method.
        """

    @abstractmethod
    def delete(self, filter=None, delete_all=False) -> dict:
        """Delete by filter (or everything when ``delete_all`` is set)."""
