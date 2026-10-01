"""Vector-store contracts, retrieval outcomes, and typed failures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

RetrievalMethod = Literal["dense", "sparse", "hybrid"]
RetrievalOutcome = Literal["success", "empty"]


@dataclass(frozen=True)
class RetrievalCandidate:
    """
    Where one candidate stood at each stage, without its text.

    A candidate is a Passage the store returned, before the app kept the five
    it would show. Its identity is the content hash and the chunk it came from,
    so two runs can be compared without either run holding the words. ``rank``
    is the position the store returned it at; ``fused_rank`` is the position it
    holds once the store's fusion has ordered the two retrievals together, and
    is None for a single-representation query where there is no fusion to
    report. ``rerank_rank`` is where the cross-encoder put it, and is None when
    the reranker did not run; ``selected`` says whether the app kept it among
    the Passages the reader is shown.
    """

    rank: int
    score: float
    content_hash: str
    chunk_index: int
    page: int | None = None
    fused_rank: int | None = None
    rerank_score: float | None = None
    rerank_rank: int | None = None
    #: Whether the app kept this candidate among the Passages it would show.
    selected: bool = False

    def to_dict(self) -> dict:
        """Return the candidate as a trace records it."""
        return {
            "rank": self.rank,
            "fused_rank": self.fused_rank,
            "rerank_rank": self.rerank_rank,
            "score": self.score,
            "rerank_score": self.rerank_score,
            "content_hash": self.content_hash,
            "chunk_index": self.chunk_index,
            "page": self.page,
            "selected": self.selected,
        }


@dataclass(frozen=True)
class RetrievalResult:
    """
    Shaped evidence plus the method and outcome that produced it.

    ``candidates`` describes what the store returned before the app bounded it,
    so an operator can see what a change to the method, the fusion, or the
    reranker did to the ranking rather than only what survived. ``rerank`` says
    whether the reranker ran for this call, with which model, and whether it
    moved anything: a rerank that left the order unchanged is a cost with no
    effect, and the trace should show that.
    """

    sources: list[dict]
    method: RetrievalMethod
    outcome: RetrievalOutcome
    candidates: tuple[RetrievalCandidate, ...] = ()
    rerank: dict | None = None


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

    def generation_report(self, filter=None, limit=1000, value_keys=()):
        """
        Describe what one index generation stores, or ``None`` when unsupported.

        A store that cannot inventory its points returns ``None``, which limits
        activation to a point-count check.
        """
        return None
