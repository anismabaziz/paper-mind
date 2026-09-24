"""Deterministic adapters and an HTTP server for full-stack tests."""

import hashlib
import math
import re
import shutil
import threading
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from qdrant_client import QdrantClient
from werkzeug.serving import make_server

import settings as settings_module
from app import create_app
from composition import Services
from repositories import build_repositories
from services.embeddings.local_embeddings import EmbeddingService
from services.llm.base import ChatCredentials, LLMProvider
from services.parsing.document_parser import DocumentIngestor
from services.retrieval.base import VectorStore
from services.retrieval.qdrant_store import QdrantIndexAdapter
from services.retrieval.reranker import Reranker
from services.retrieval.vector_service import VectorService
from settings import Settings
from storage import LocalStorage

_EMBEDDING_SIZE = 1024


class InjectedFailure(RuntimeError):
    """A deterministic failure requested by a full-stack test."""


def _tokens(text: str) -> list[str]:
    """Split text into stable lowercase terms."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _maybe_fail(failures: set[str], operation: str) -> None:
    """Raise when an operation has a requested failure."""
    if operation in failures:
        raise InjectedFailure(f"{operation} unavailable")


class ControlledStorage(LocalStorage):
    """Local storage that can fail one public operation on demand."""

    def __init__(self, root: Path):
        """Bind a real local store and start without failures."""
        super().__init__(root)
        self.root = root
        self.failures: set[str] = set()

    def fail(self, operation: str) -> None:
        """Fail the named operation on its next call."""
        self.failures.add(operation)

    def unfail(self, operation: str) -> None:
        """Clear a requested failure so a retry can succeed."""
        self.failures.discard(operation)

    def save(self, filename: str, content: bytes) -> None:
        """Store bytes unless the save failure is active."""
        _maybe_fail(self.failures, "save")
        super().save(filename, content)

    def open(self, filename: str) -> bytes:
        """Read bytes unless the open failure is active."""
        _maybe_fail(self.failures, "open")
        return super().open(filename)

    def exists(self, filename: str) -> bool:
        """Check existence unless the exists failure is active."""
        _maybe_fail(self.failures, "exists")
        return super().exists(filename)

    def delete(self, filename: str) -> None:
        """Delete bytes unless the delete failure is active."""
        _maybe_fail(self.failures, "delete")
        super().delete(filename)

    def list(self) -> list[dict]:
        """List stored files unless the list failure is active."""
        _maybe_fail(self.failures, "list")
        return super().list()

    def url(self, filename: str) -> str:
        """Return the download path for a stored file."""
        return super().url(filename)


class ControlledRepository:
    """Repository proxy that can fail one public method on demand."""

    def __init__(self, delegate):
        """Wrap a real repository."""
        self._delegate = delegate
        self.failures: set[str] = set()

    def fail(self, operation: str) -> None:
        """Fail the named repository method on its next call."""
        self.failures.add(operation)

    def unfail(self, operation: str) -> None:
        """Clear a requested failure so a retry can succeed."""
        self.failures.discard(operation)

    def __getattr__(self, name: str):
        """Delegate a repository method after checking its failure switch."""
        method = getattr(self._delegate, name)

        def invoke(*args, **kwargs):
            _maybe_fail(self.failures, name)
            return method(*args, **kwargs)

        return invoke


@dataclass(frozen=True)
class ControlledRepositories:
    """Repository proxies used by the HTTP routes."""

    files: ControlledRepository
    app_settings: ControlledRepository
    conversations: ControlledRepository


class ControlledEmbeddingService(EmbeddingService):
    """Free 1024-dimensional embeddings with recorded inputs and failures."""

    def __init__(self):
        """Start with recorded calls and no active failure."""
        self.calls: list[list[str]] = []
        self.failures: set[str] = set()

    def fail(self, operation: str) -> None:
        """Fail the named embedding operation on its next call."""
        self.failures.add(operation)

    def embed_texts(self, texts):
        """Return normalized feature-hashed vectors in input order."""
        _maybe_fail(self.failures, "embed")
        values = [texts] if isinstance(texts, str) else list(texts)
        self.calls.append(values)
        return [self._embed(value) for value in values]

    @staticmethod
    def _embed(text: str) -> list[float]:
        """Hash terms into one stable unit vector."""
        vector = [0.0] * _EMBEDDING_SIZE
        for token, frequency in Counter(_tokens(text)).items():
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % _EMBEDDING_SIZE
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * frequency
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]


class ControlledReranker(Reranker):
    """Deterministic lexical reranker with recorded inputs and failures."""

    def __init__(self):
        """Start with recorded calls and no active failure."""
        self.calls: list[tuple[str, list[dict]]] = []
        self.failures: set[str] = set()

    def fail(self, operation: str) -> None:
        """Fail the named reranker operation on its next call."""
        self.failures.add(operation)

    def maybe_rerank(
        self, query: str | None, sources: list[dict], enabled: bool | None = None
    ) -> list[dict]:
        """Order sources by query-term overlap unless failure is active."""
        self.calls.append((query or "", [dict(source) for source in sources]))
        _maybe_fail(self.failures, "rerank")
        if query is None or enabled is False:
            return sources
        query_terms = set(_tokens(query))
        ranked = []
        for source in sources:
            overlap = len(query_terms.intersection(_tokens(source.get("content", ""))))
            ranked.append({**source, "score": float(overlap), "rerank_score": overlap})
        ranked.sort(key=lambda source: source["score"], reverse=True)
        return ranked


class ControlledVectorStore(VectorStore):
    """Real Qdrant adapter that can fail one public operation on demand."""

    def __init__(self, delegate: QdrantIndexAdapter):
        """Wrap the production Qdrant adapter."""
        self._delegate = delegate
        self.failures: set[str] = set()

    def fail(self, operation: str) -> None:
        """Fail the named vector operation on its next call."""
        self.failures.add(operation)

    def unfail(self, operation: str) -> None:
        """Clear a requested failure so a retry can succeed."""
        self.failures.discard(operation)

    def upsert(self, vectors) -> dict:
        """Store vectors unless the upsert failure is active."""
        _maybe_fail(self.failures, "upsert")
        return self._delegate.upsert(vectors)

    def query(self, vector, top_k, include_metadata=True, filter=None, **kwargs):
        """Query vectors unless the query failure is active."""
        _maybe_fail(self.failures, "query")
        return self._delegate.query(
            vector,
            top_k,
            include_metadata=include_metadata,
            filter=filter,
            **kwargs,
        )

    def delete(self, filter=None, delete_all=False) -> dict:
        """Delete vectors unless the delete failure is active."""
        _maybe_fail(self.failures, "delete")
        return self._delegate.delete(filter=filter, delete_all=delete_all)


class DeterministicChatProvider(LLMProvider):
    """Chat provider that streams fixed text without a network client."""

    name = "deterministic"

    def __init__(
        self, factory: "DeterministicChatFactory", credentials: ChatCredentials
    ):
        """Bind the provider to the test factory and stored credentials."""
        super().__init__(credentials.api_key, credentials.model)
        self._factory = factory

    def _build_client(self):
        """Return no client because this provider has no SDK."""
        return None

    def _generate_response(self, query: str, context: str) -> str:
        """Return the fixed answer after recording its inputs."""
        _maybe_fail(self._factory.failures, "generate")
        self._factory.generated.append((query, context))
        return "The answer is grounded in the retrieved PDF text."

    def _stream_response(self, query: str, context: str) -> Iterator[str]:
        """Yield the fixed answer after recording its inputs."""
        _maybe_fail(self._factory.failures, "stream")
        self._factory.streamed.append((query, context))
        yield "The answer is "
        yield "grounded in the retrieved PDF text."

    def verify(self) -> None:
        """Verify the deterministic provider without an external call."""
        _maybe_fail(self._factory.failures, "verify")
        self._factory.verified.append((self.api_key, self.model))


@dataclass
class DeterministicChatFactory:
    """Provider factory that records credentials and can fail generation."""

    built: list[LLMProvider] = field(default_factory=list)
    credentials: list[ChatCredentials] = field(default_factory=list)
    generated: list[tuple[str, str]] = field(default_factory=list)
    streamed: list[tuple[str, str]] = field(default_factory=list)
    verified: list[tuple[str, str]] = field(default_factory=list)
    failures: set[str] = field(default_factory=set)

    def fail(self, operation: str) -> None:
        """Fail the named provider operation on its next call."""
        self.failures.add(operation)

    def __call__(self, credentials: ChatCredentials) -> LLMProvider:
        """Build the deterministic provider for stored credentials."""
        self.credentials.append(credentials)
        provider = DeterministicChatProvider(self, credentials)
        self.built.append(provider)
        return provider

    def verify(self, credentials: ChatCredentials) -> tuple[bool, str | None]:
        """Verify candidate credentials through the provider interface."""
        try:
            self(credentials).verify()
        except Exception as exc:
            return False, str(exc)
        return True, None


@dataclass
class ApplicationHarness:
    """Running HTTP application and direct access to its real dependencies."""

    client: httpx.Client
    server: Any
    server_thread: threading.Thread
    session_factory: Any
    qdrant: QdrantClient
    collection_name: str
    storage: ControlledStorage
    repositories: ControlledRepositories
    embeddings: ControlledEmbeddingService
    reranker: ControlledReranker
    vector_store: ControlledVectorStore
    chat: DeterministicChatFactory

    def close(self) -> None:
        """Stop HTTP serving and remove Qdrant and file-system test data."""
        self.client.close()
        self.server.shutdown()
        self.server_thread.join(timeout=5)
        try:
            if self.qdrant.collection_exists(self.collection_name):
                self.qdrant.delete_collection(self.collection_name)
        finally:
            self.qdrant.close()
            shutil.rmtree(self.storage.root)


def build_application(
    app_settings: Settings,
    session_factory,
) -> ApplicationHarness:
    """Serve the production app with real infrastructure and local adapters."""
    settings_module.set_settings(app_settings)
    qdrant = QdrantClient(url=app_settings.vector.qdrant_url, timeout=30)
    storage = ControlledStorage(app_settings.storage.storage_dir)
    repositories = build_repositories(session_factory)
    controlled_repositories = ControlledRepositories(
        files=ControlledRepository(repositories.files),
        app_settings=ControlledRepository(repositories.app_settings),
        conversations=ControlledRepository(repositories.conversations),
    )
    embeddings = ControlledEmbeddingService()
    reranker = ControlledReranker()
    vector_store = ControlledVectorStore(
        QdrantIndexAdapter(qdrant, app_settings.vector.index_name)
    )
    chat = DeterministicChatFactory()
    services = Services(
        settings=app_settings,
        repositories=controlled_repositories,
        storage=storage,
        parser=DocumentIngestor(
            app_settings.parsing.use_docling,
            app_settings.chunking.chunk_size_tokens,
            app_settings.chunking.chunk_overlap_tokens,
        ),
        embedding_service=embeddings,
        vector_service=VectorService(vector_store, reranker),
        chat_provider_factory=chat,
        api_key_verifier=chat.verify,
    )
    application = create_app(app_settings, services=services)
    server = make_server("127.0.0.1", 0, application, threaded=True)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    client = httpx.Client(base_url=f"http://127.0.0.1:{server.server_port}", timeout=30)
    for _ in range(50):
        try:
            if client.get("/health").status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    else:
        client.close()
        server.shutdown()
        server_thread.join(timeout=5)
        qdrant.close()
        raise RuntimeError("full-stack HTTP application did not become ready")

    return ApplicationHarness(
        client=client,
        server=server,
        server_thread=server_thread,
        session_factory=session_factory,
        qdrant=qdrant,
        collection_name=app_settings.vector.index_name,
        storage=storage,
        repositories=controlled_repositories,
        embeddings=embeddings,
        reranker=reranker,
        vector_store=vector_store,
        chat=chat,
    )
