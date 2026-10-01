"""
Deterministic collaborators for the evaluation tests.

A run is only worth reading if it describes the application, so the tests
build the real thing: real repositories over an in-memory database, the real
parser, the real ingestion worker, the real vector service, and the real answer
path. What is substituted are the three things that would otherwise need a
model server, a vector database, and a paid provider — and each substitute
speaks the same interface as the thing it stands in for, including the sparse
and hybrid parts of the retrieval contract.
"""

import hashlib
import math
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from services.llm.base import (
    ChatBudget,
    ChatCredentials,
    LLMProvider,
    ProviderTimeoutError,
)
from services.citations import claims_block
from services.retrieval.base import VectorStore, VectorStoreConfigurationError
from services.retrieval.hybrid import RRF_K, build_sparse_vector
from storage import LocalStorage

EMBEDDING_SIZE = 1024

STOPWORDS = frozenset(
    """a an and are as at be by for from has have how in is it its of on or that
    the this to was were what where which with""".split()
)


def tokens(text: str) -> list[str]:
    """Split text into the content words retrieval and embedding both use."""
    return [
        token
        for token in re.findall(r"[a-z0-9%°]+", text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


def embed(text: str) -> list[float]:
    """Return one stable unit vector for a piece of text."""
    vector = [0.0] * EMBEDDING_SIZE
    for token, frequency in Counter(tokens(text)).items():
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % EMBEDDING_SIZE
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign * frequency
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        vector[0] = 1.0
        return vector
    return [value / norm for value in vector]


class HashingEmbeddingService:
    """Embeds text with the deterministic hash above, in input order."""

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.embedded: list[list[str]] = []

    def embed_texts(self, texts, check=None):
        """Return one vector per input text."""
        values = [texts] if isinstance(texts, str) else list(texts)
        self.embedded.append(values)
        return [embed(value) for value in values]


class InMemoryVectorStore(VectorStore):
    """
    A vector store with the whole retrieval contract.

    Dense, sparse, and hybrid queries all work, the dense and sparse vectors a
    point is stored with are both kept, and a query that cannot be served
    raises rather than quietly answering with something else.
    """

    def __init__(self, supports_hybrid: bool = True) -> None:
        """Start empty; optionally without the ability to serve hybrid queries."""
        self.points: list[dict[str, Any]] = []
        self.supports_hybrid = supports_hybrid

    def upsert(self, vectors) -> dict:
        """Store the given points, replacing any with the same id."""
        stored = {point["id"] for point in vectors}
        self.points = [point for point in self.points if point["id"] not in stored]
        self.points.extend(vectors)
        return {"status": "completed"}

    def _selected(self, point_filter):
        return [
            point
            for point in self.points
            if all(
                point["metadata"].get(key) == value
                for key, value in (point_filter or {}).items()
            )
        ]

    def query(self, vector, top_k, include_metadata=True, filter=None, **kwargs):
        """Answer one dense, sparse, or hybrid query, or fail loudly."""
        method = kwargs.get("method", "dense")
        sparse_vector = kwargs.get("sparse_vector")
        if method not in {"dense", "sparse", "hybrid"}:
            raise VectorStoreConfigurationError(
                f"Unsupported retrieval method: {method}"
            )
        if method != "dense" and not self.supports_hybrid:
            raise VectorStoreConfigurationError(
                "this store cannot serve sparse or hybrid queries"
            )
        if method != "dense" and sparse_vector is None:
            raise VectorStoreConfigurationError(
                f"{method} retrieval requires a sparse query vector"
            )
        candidates = self._selected(filter)
        dense = self._ranked(candidates, vector, top_k, self._cosine)
        if method == "dense":
            matches = dense
        else:
            sparse = self._ranked(
                candidates, sparse_vector, top_k, self._sparse_dot, sparse=True
            )
            matches = sparse if method == "sparse" else self._fuse(dense, sparse)
        return {
            "matches": matches[:top_k],
            "method": method,
            "outcome": "success" if matches else "empty",
        }

    def count(self, filter=None) -> int:
        """Count the points matching a filter."""
        return len(self._selected(filter))

    def delete(self, filter=None, delete_all=False) -> dict:
        """Delete the points matching a filter, or all of them."""
        if delete_all:
            self.points = []
        else:
            kept = {point["id"] for point in self._selected(filter)}
            self.points = [point for point in self.points if point["id"] not in kept]
        return {"status": "completed"}

    @staticmethod
    def _ranked(points, query, top_k, score, sparse=False):
        matches = []
        for point in points:
            stored = (
                point.get("sparse_vector")
                or build_sparse_vector(point["metadata"]["content"])
                if sparse
                else point["values"]
            )
            value = score(query, stored)
            if value > 0:
                matches.append(
                    {
                        "id": point["id"],
                        "score": value,
                        "metadata": dict(point["metadata"]),
                    }
                )
        matches.sort(key=lambda match: match["score"], reverse=True)
        return matches

    @staticmethod
    def _cosine(left, right) -> float:
        dot = sum(x * y for x, y in zip(left, right))
        norm = math.sqrt(sum(x * x for x in left)) * math.sqrt(
            sum(x * x for x in right)
        )
        return dot / norm if norm else 0.0

    @staticmethod
    def _sparse_dot(query, document) -> float:
        query_values = dict(zip(query["indices"], query["values"]))
        document_values = dict(zip(document["indices"], document["values"]))
        return sum(
            value * document_values.get(index, 0.0)
            for index, value in query_values.items()
        )

    @staticmethod
    def _fuse(dense, sparse):
        fused: dict[str, dict] = {}
        for matches in (dense, sparse):
            for rank, match in enumerate(matches, start=1):
                entry = fused.setdefault(match["id"], {**match, "score": 0.0})
                entry["score"] += 1.0 / (RRF_K + rank)
        return sorted(fused.values(), key=lambda match: match["score"], reverse=True)


class InMemoryStorage(LocalStorage):
    """Local storage held in a directory the test removes afterwards."""

    def __init__(self, root) -> None:
        """Start with nothing stored."""
        super().__init__(root)
        self.blobs: dict[str, bytes] = {}

    def save(self, filename: str, content: bytes) -> None:
        """Store bytes in memory and on disk."""
        self.blobs[filename] = content
        super().save(filename, content)

    def open(self, filename: str) -> bytes:
        """Read stored bytes."""
        return self.blobs[filename]


@dataclass
class ScriptedPlan:
    """What one stream does, so a test chooses the outcome it needs."""

    fragments: list[str] = field(default_factory=list)
    error: BaseException | None = None
    before_output: bool = False
    finish_reason: str | None = "stop"


class ScriptedProvider(LLMProvider):
    """Streams a scripted answer, citing a Passage the context supplied."""

    name = "scripted"

    def __init__(self, factory: "ScriptedChatFactory", credentials: ChatCredentials):
        """Bind the provider to the script and the budget it was given."""
        super().__init__(
            credentials.api_key,
            credentials.model,
            budget=credentials.budget or ChatBudget(),
        )
        self._factory = factory

    def _build_client(self):
        """No SDK client is needed for a scripted answer."""
        return None

    def verify(self) -> None:
        """Verification is not exercised over the answer path."""
        return None

    def _generate_response(self, query: str, context: str, history: str = "") -> str:
        """Answer a one-shot call, such as a citation repair, from the script."""
        self._factory.completed.append((query, context, history))
        return self._factory.repair

    def _stream_response(
        self, query: str, context: str, history: str = ""
    ) -> Iterator[str]:
        """Yield the scripted answer, raising the scripted failure where asked."""
        plan = self._factory.next_plan()
        self._factory.streamed.append((query, context, history))
        if plan.error is not None and plan.before_output:
            raise plan.error
        yield from plan.fragments
        if plan.error is not None:
            raise plan.error
        self.last_finish_reason = plan.finish_reason


@dataclass
class ScriptedChatFactory:
    """Builds scripted providers and records what they were asked."""

    streamed: list[tuple[str, str, str]] = field(default_factory=list)
    completed: list[tuple[str, str, str]] = field(default_factory=list)
    plans: list[ScriptedPlan] = field(default_factory=list)
    repair: str = ""
    attempts: int = 0

    def __call__(self, credentials: ChatCredentials) -> LLMProvider:
        """Build a provider under the credentials the answer path resolved."""
        return ScriptedProvider(self, credentials)

    def next_plan(self) -> ScriptedPlan:
        """Return what the next stream does, repeating the last plan."""
        index = min(self.attempts, len(self.plans) - 1)
        self.attempts += 1
        return self.plans[index]

    def answer(self, text: str, *, cite: bool = True) -> None:
        """Answer the next question with this text and a valid claims block."""
        self.plans = [ScriptedPlan(fragments=[text, self._claims(text, cite)])]

    def answer_with_claims(self, text: str, claims: list[dict[str, Any]]) -> None:
        """Answer with claims the test wrote, so a grader's input is chosen."""
        self.plans = [
            ScriptedPlan(
                fragments=[text, f"<claims>\n{claims_block(claims)}\n</claims>"]
            )
        ]

    def fail_with(self, error: BaseException, *, before_output: bool = False) -> None:
        """Make the next stream raise, before or after any visible output."""
        self.plans = [
            ScriptedPlan(fragments=[], error=error, before_output=before_output)
        ]

    def answer_without_citations(self, text: str) -> None:
        """Answer with prose and no claims block, so no claim can be checked."""
        self.plans = [ScriptedPlan(fragments=[text])]

    @staticmethod
    def _claims(text: str, cite: bool) -> str:
        """Return a claims block citing a Passage the context actually gave."""
        if not cite:
            return ""
        return (
            '<claims>\n{"claim": "the grounded statement", "sources": ["S1"]}\n'
            "</claims>"
        )

    def uncited(self, text: str) -> None:
        """Answer with a claim citing a Passage that was never supplied."""
        self.plans = [
            ScriptedPlan(
                fragments=[
                    text,
                    '<claims>\n{"claim": "invented", "sources": ["S99"]}\n</claims>',
                ]
            )
        ]


def timeout_error() -> ProviderTimeoutError:
    """Return the failure a stalled provider raises."""
    return ProviderTimeoutError("scripted timeout")


def in_memory_sessions():
    """Return a session factory over a fresh in-memory database."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from db import Base

    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def build_environment_for(
    dataset,
    settings_obj,
    storage_root,
    *,
    store=None,
    prefix: str = "",
    chat_factory=None,
):
    """
    Index the case set's documents into a working application.

    The evaluation tests all need the same thing: the real repositories, the
    real ingestion worker, and the real retrieval service over a store that
    keeps both vector representations, with only the embedding weights, the
    vector database, and the provider substituted.
    """
    from evaluation.harness import build_environment
    from services.retrieval.vector_service import VectorService

    return build_environment(
        dataset,
        settings=settings_obj,
        session_factory=in_memory_sessions(),
        storage=InMemoryStorage(storage_root),
        embedding_service=HashingEmbeddingService(),
        vector_service=VectorService(store or InMemoryVectorStore()),
        chat_provider_factory=chat_factory or ScriptedChatFactory(),
        documents_prefix=prefix,
    )
