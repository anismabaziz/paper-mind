"""
A composed app for the chat boundary, with every collaborator faked.

The chat tests all need the same thing: real repositories over an in-memory
database, a stored provider key, one indexed Document to ask about, and a
chat provider whose stream the test scripts. Building that once keeps the
event-contract tests about the protocol rather than about the harness.
"""

import io
from dataclasses import dataclass, field, replace
from typing import Any, Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import create_app
from composition import Services
from db import Base
from repositories import build_repositories
from services.accounts.secrets_service import encrypt_api_key
from services.llm.base import ChatCredentials, LLMProvider
from services.parsing.document_parser import Chunk
from services.retrieval.base import RetrievalResult
from services.retrieval.vector_service import FETCH_K
from tests.ingestion_helpers import build_test_worker

CHUNK = Chunk(text="chunk about topic", page_no=1, chunk_index=0, content_hash="hash-0")
DEFAULT_MATCH = {
    "content": "chunk about topic",
    "document": "doc.pdf",
    "chunk_index": 0,
    "score": 0.9,
    "page": 1,
}


def in_memory_session_factory():
    """Build a session factory over one in-memory database."""
    engine = create_engine(
        "sqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


class _Storage:
    """In-memory stand-in for the storage layer."""

    def __init__(self):
        """Start with no stored bytes."""
        self.blobs: dict[str, bytes] = {}

    def save(self, filename, content):
        """Store bytes."""
        self.blobs[filename] = content

    def open(self, filename):
        """Read bytes."""
        return self.blobs[filename]

    def exists(self, filename):
        """Report whether bytes are stored."""
        return filename in self.blobs

    def delete(self, filename):
        """Remove bytes."""
        self.blobs.pop(filename, None)

    def list(self):
        """List stored files."""
        return [
            {"name": name, "size": len(data)}
            for name, data in sorted(self.blobs.items())
        ]

    def url(self, filename):
        """Return the download path for a stored file."""
        return f"/storage/{filename}"


class _Vectors:
    """Scripted-match stand-in for the vector service."""

    def __init__(self):
        """Start with one grounded match, or whatever a test scripts in."""
        self.matches: list[dict[str, Any]] = [DEFAULT_MATCH]
        self.raise_with: BaseException | None = None

    def script(self, *matches: dict[str, Any]) -> None:
        """Return exactly these matches for the next questions."""
        self.matches = list(matches)

    def fail_with(self, error: BaseException) -> None:
        """Make the next retrieval raise, as an unreachable store would."""
        self.raise_with = error

    def upsert_chunks(self, embeddings, chunks, filename, **kwargs):
        """Accept an upsert without storing anything."""

    def query_vectors(
        self, embedding, filename, top_k=FETCH_K, query_text=None, **kwargs
    ):
        """Return the scripted matches, or raise what the test scripted."""
        if self.raise_with is not None:
            raise self.raise_with
        return RetrievalResult(
            sources=[dict(match) for match in self.matches],
            method="hybrid",
            outcome="success" if self.matches else "empty",
        )

    def delete_by_filename(self, filename):
        """Accept a delete without storing anything."""

    def delete_all(self):
        """Accept a bulk delete without storing anything."""


class _Parser:
    """Single-chunk stand-in for the parse-and-chunk pipeline."""

    def get_chunk_objects(self, filename, file_bytes):
        """Return one chunk."""
        return [CHUNK]


class _Embeddings:
    """Fixed-vector stand-in."""

    def embed_texts(self, texts):
        """Return one vector per text."""
        if isinstance(texts, str):
            texts = [texts]
        return [[0.1, 0.2] for _ in texts]


@dataclass
class StreamPlan:
    """Script one attempt of a chat stream, so a test can choose the outcome."""

    fragments: list[str] = field(default_factory=lambda: ["The answer ", "is 42."])
    error: BaseException | None = None
    before_output: bool = False
    finish_reason: str | None = "stop"


class _ChatProvider(LLMProvider):
    """
    Streams exactly what the plan on its factory says.

    It is a real provider, not a stub, so the route's stream goes through the
    same budgeting, retry, and finish-reason handling the SDK providers get.
    """

    name = "fake"

    def __init__(self, factory):
        """Bind the provider to its factory and the budget it was given."""
        super().__init__(
            "sk-test-chat-key",
            factory.credentials.model,
            budget=factory.credentials.budget,
        )
        self._factory = factory

    def _build_client(self):
        """No SDK client is needed for a scripted stream."""
        return None

    def verify(self) -> None:
        """Key verification is not exercised over the chat boundary."""
        raise NotImplementedError

    def _generate_response(self, query, context, prior_turns="") -> str:
        """Answer a one-shot call, such as a citation repair, from the script."""
        self._factory.completed.append((query, context, prior_turns))
        index = min(self._factory.completions_used, len(self._factory.completions) - 1)
        self._factory.completions_used += 1
        return self._factory.completions[index]

    def _stream_response(self, query, context, prior_turns="") -> Iterator[str]:
        """Yield the planned fragments, raising the planned error where asked."""
        plan = self._factory.plan_for(self._factory.attempts)
        self._factory.attempts += 1
        if plan.error is not None and plan.before_output:
            raise plan.error
        self._factory.streamed.append((query, context, prior_turns))
        yield from plan.fragments
        if plan.error is not None:
            raise plan.error
        self.last_finish_reason = plan.finish_reason


class _ChatFactory:
    """Builds a streaming provider whose next stream follows its plan."""

    def __init__(self):
        """Start with a plan that answers normally."""
        self.streamed: list[tuple[str, str, str]] = []
        self.completed: list[tuple[str, str, str]] = []
        self.plans: list[StreamPlan] = [StreamPlan()]
        self.attempts = 0
        self.credentials: ChatCredentials | None = None
        self.completions: list[str] = [""]
        self.completions_used = 0

    def __call__(self, credentials):
        """Build a streaming provider under the credentials the route resolved."""
        self.credentials = credentials
        return _ChatProvider(self)

    def plan_for(self, attempt: int) -> StreamPlan:
        """Return what attempt number N of the next stream does."""
        index = min(attempt, len(self.plans) - 1)
        return self.plans[index]

    def stream(self, *fragments: str) -> None:
        """Answer the next question with exactly these fragments."""
        self.plans = [StreamPlan(fragments=list(fragments))]

    def fail_with(self, error: BaseException, *, before_output: bool = False) -> None:
        """Make the next stream raise, before or after any visible output."""
        self.plans = [
            StreamPlan(fragments=[], error=error, before_output=before_output)
        ]

    def script(self, *plans: StreamPlan) -> None:
        """
        Script successive attempts of the next stream.

        The last plan repeats, so a script that only says "the first attempt
        fails" describes a transient failure followed by a working provider.
        """
        self.plans = list(plans)

    def complete_with(self, *replies: str) -> None:
        """
        Script what one-shot calls answer, such as a citation repair.

        The last reply repeats, so a script that gives one repair covers both
        "the first repair fixes it" and "every repair fails".
        """
        self.completions = list(replies)
        self.completions_used = 0


@pytest.fixture
def session_factory():
    """Session factory over one in-memory database."""
    return in_memory_session_factory()


@pytest.fixture
def repositories(session_factory):
    """Build every repository over one in-memory database with saved settings."""
    repositories = build_repositories(session_factory)
    repositories.app_settings.upsert_app_settings(
        "groq", "openai/gpt-oss-120b", encrypt_api_key("sk-test-chat-key")
    )
    return repositories


@pytest.fixture
def fake_chat():
    """Chat provider factory whose next stream the test scripts."""
    return _ChatFactory()


@pytest.fixture
def app(repositories, fake_chat, settings_obj):
    """Compose the app with fakes over the in-memory database."""
    storage = _Storage()
    services = replace(
        Services.from_settings(settings_obj),
        repositories=repositories,
        storage=storage,
        parser=_Parser(),
        embedding_service=_Embeddings(),
        vector_service=_Vectors(),
        chat_provider_factory=fake_chat,
    )
    application = create_app(settings_obj, services=services)
    application.config.update(
        TEST_REPOSITORIES=repositories,
        TEST_STORAGE=storage,
        TEST_PARSER=services.parser,
        TEST_EMBEDDINGS=services.embedding_service,
        TEST_VECTORS=services.vector_service,
    )
    return application


@pytest.fixture
def client(app):
    """HTTP client for the composed app."""
    return app.test_client()


def indexed_document(client, app, name="doc.pdf") -> str:
    """Upload a document and index it so chat can be asked a question."""
    data = {"file": (io.BytesIO(b"%PDF-fake-bytes"), name)}
    filename = client.post(
        "/upload", data=data, content_type="multipart/form-data"
    ).get_json()["file"]["name"]
    assert client.post("/process-file", json={"filename": filename}).status_code == 202
    assert build_test_worker(app, "test-worker").drain() == 1
    return filename


def turns_of(repositories: Any, filename: str) -> list[dict]:
    """Read one document's conversation turns in sequence order."""
    file_id = repositories.files.get_file(filename)["id"]
    conversation_id = repositories.conversations.get_conversation_id(file_id)
    return repositories.conversations.get_turns(conversation_id)
