"""
Service-layer flow tests.

This module keeps every external edge fake: vector index, embeddings, parser,
LLM, and storage are constructed here and injected through ``create_app``;
the repository runs against in-memory SQLite.
"""

import ast
import io
import logging
import os
import pathlib
import subprocess
import sys
from dataclasses import replace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import create_app
from composition import Services
from db import AppSettings, Base
from repositories import build_repositories
from services.accounts.secrets_service import encrypt_api_key
from services.llm.base import ChatCredentials, LLMProvider
from services.parsing.document_parser import Chunk
from services.retrieval.base import (
    RetrievalResult,
    VectorDimensionError,
    VectorStoreConfigurationError,
    VectorStoreUnavailableError,
)
from services.retrieval.hybrid import build_sparse_vector
from services.retrieval.vector_service import FETCH_K, shape_sources
from tests.ingestion_helpers import build_test_worker
from tests.sse import parse_sse


@pytest.fixture
def repositories():
    """Build all aggregate repositories over one in-memory database."""
    engine = create_engine(
        "sqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    repositories = build_repositories(sessionmaker(bind=engine))
    repositories.app_settings.upsert_app_settings(
        "groq", "openai/gpt-oss-120b", encrypt_api_key("sk-test-chat-key")
    )
    return repositories


class FakeStorage:
    """In-memory stand-in for the storage layer."""

    def __init__(self):
        """Initialize."""
        self.blobs = {}

    def save(self, filename, content):
        """Do save."""
        self.blobs[filename] = content

    def open(self, filename):
        """Do open."""
        return self.blobs[filename]

    def exists(self, filename):
        """Do exists."""
        return filename in self.blobs

    def delete(self, filename):
        """Do delete."""
        self.blobs.pop(filename, None)

    def list(self):
        """Do list."""
        return [
            {"name": name, "size": len(data)}
            for name, data in sorted(self.blobs.items())
        ]

    def url(self, filename):
        """Do url."""
        return f"/storage/{filename}"


@pytest.fixture
def fake_storage():
    """Fresh in-memory storage, injected through the app factory."""
    return FakeStorage()


class FakeVectorService:
    """FakeVectorService."""

    def __init__(self):
        """Initialize."""
        self.upserts = []
        self.deleted = []
        self.deleted_all = False
        self.matches = [
            {
                "content": "chunk about topic",
                "document": "doc.pdf",
                "chunk_index": 0,
                "score": 0.92,
            }
        ]

    def upsert_chunks(self, embeddings, chunks, filename, **kwargs):
        """Do upsert chunks."""
        self.upserts.append((embeddings, chunks, filename))

    def query_vectors(
        self, embedding, filename, top_k=FETCH_K, query_text=None, **kwargs
    ):
        # Route through the real shaping so flow tests see the same
        # dedupe/bound/order behavior as production retrieval.
        """Do query vectors."""
        sources = shape_sources(self.matches)
        method = (
            "hybrid"
            if query_text and build_sparse_vector(query_text)["indices"]
            else "dense"
        )
        return RetrievalResult(
            sources=sources,
            method=method,
            outcome="success" if sources else "empty",
        )

    def delete_by_filename(self, filename):
        """Do delete by filename."""
        self.deleted.append(filename)

    def delete_all(self):
        """Do delete all."""
        self.deleted_all = True


@pytest.fixture
def fake_vectors():
    """Fresh fake vector service, injected through the app factory."""
    return FakeVectorService()


class FakeParser:
    """Fixed-chunk stand-in for the parse-and-chunk pipeline."""

    def get_chunk_objects(self, filename, file_bytes):
        """Do get chunk objects."""
        return [
            Chunk(
                text="chunk about topic",
                page_no=1,
                chunk_index=0,
                content_hash="hash-0",
            ),
            Chunk(
                text="another chunk",
                page_no=1,
                chunk_index=1,
                content_hash="hash-1",
            ),
        ]


class FakeEmbeddingService:
    """Fixed-vector stand-in that records every embedded text."""

    def __init__(self):
        """Initialize."""
        self.embedded = []

    def embed_texts(self, texts):
        """Do embed texts."""
        if isinstance(texts, str):
            texts = [texts]
        self.embedded.extend(texts)
        return [[0.1, 0.2] for _ in texts]


class FakeChatProvider:
    """Streams fixed tokens through its factory so tests can observe calls."""

    def __init__(self, factory):
        """Initialize."""
        self._factory = factory

    def stream_response(self, query, context, history=""):
        """Do stream response."""
        if self._factory.provider_error is not None:
            raise self._factory.provider_error
        self._factory.streamed.append((query, context, history))
        yield "The answer "
        yield "is 42."

    def generate_response(self, query, context, history=""):
        """Do generate response."""
        self._factory.answered.append((query, context, history))
        return "The answer is 42."


class FakeChatFactory:
    """
    Fake chat provider factory.

    Records the credentials each build received and the streamed turns;
    setting ``provider_error`` makes the built provider raise on its next
    stream, as if the provider were down.
    """

    def __init__(self):
        """Initialize."""
        self.built = None
        self.streamed = []
        self.answered = []
        self.provider_error = None

    def __call__(self, credentials):
        """Do build provider."""
        self.built = credentials
        return FakeChatProvider(self)


@pytest.fixture
def fake_parser():
    """Fresh fake parser, injected through the app factory."""
    return FakeParser()


@pytest.fixture
def fake_embeddings():
    """Fresh fake embedding service, injected through the app factory."""
    return FakeEmbeddingService()


@pytest.fixture
def fake_chat():
    """Fresh fake chat provider factory, injected through the app factory."""
    return FakeChatFactory()


@pytest.fixture
def app(
    repositories,
    fake_storage,
    fake_vectors,
    fake_parser,
    fake_embeddings,
    fake_chat,
    settings_obj,
):
    """Compose the app like production, with fakes wired through the factory."""
    services = replace(
        Services.from_settings(settings_obj),
        repositories=repositories,
        storage=fake_storage,
        parser=fake_parser,
        embedding_service=fake_embeddings,
        vector_service=fake_vectors,
        chat_provider_factory=fake_chat,
    )
    application = create_app(settings_obj, services=services)
    application.config.update(
        TEST_REPOSITORIES=repositories,
        TEST_STORAGE=fake_storage,
        TEST_PARSER=fake_parser,
        TEST_EMBEDDINGS=fake_embeddings,
        TEST_VECTORS=fake_vectors,
    )
    return application


@pytest.fixture
def client(app):
    """Do client."""
    with app.test_client() as client:
        yield client


def upload(client, name="doc.pdf"):
    """Do upload."""
    data = {"file": (io.BytesIO(b"%PDF-fake-bytes"), name)}
    return client.post("/upload", data=data, content_type="multipart/form-data")


def run_ingestion(app, worker_id="test-worker"):
    """Drain queued ingestion jobs through the worker."""
    return build_test_worker(app, worker_id).drain()


def index_document(client, app, name="doc.pdf"):
    """Upload a document and run its durable ingestion job to ready."""
    filename = upload(client, name).get_json()["file"]["name"]
    assert client.post("/process-file", json={"filename": filename}).status_code == 202
    assert run_ingestion(app) == 1
    return filename


def test_upload_stores_bytes_and_creates_record(client, fake_storage):
    """Do test upload stores bytes and creates record."""
    response = upload(client)

    assert response.status_code == 200
    body = response.get_json()
    filename = body["file"]["name"]
    assert fake_storage.blobs[filename] == b"%PDF-fake-bytes"
    assert body["file"]["url"] == f"http://localhost/storage/{filename}"

    listing = client.get("/files").get_json()["files"]
    assert [f["name"] for f in listing] == [filename]
    assert listing[0]["metadata"]["size"] == len(b"%PDF-fake-bytes")
    assert listing[0]["is_processed"] is False


def test_upload_without_file_is_rejected(client):
    """Do test upload without file is rejected."""
    assert client.post("/upload", data={}).status_code == 400


def test_process_embeds_and_marks_processed(client, app, fake_vectors, fake_embeddings):
    """Do test process embeds and marks processed."""
    filename = upload(client).get_json()["file"]["name"]

    response = client.post("/process-file", json={"filename": filename})

    assert response.status_code == 202
    assert response.get_json()["job"]["state"] == "queued"
    assert fake_vectors.upserts == [], "the request must not index inline"
    assert fake_embeddings.embedded == []

    assert run_ingestion(app) == 1
    assert fake_vectors.upserts, "document should reach the vector index"
    _, texts, upserted_file = fake_vectors.upserts[0]
    assert upserted_file == filename
    assert texts, "chunks should be extracted before embedding"
    assert fake_embeddings.embedded, "chunks should be embedded"
    assert (
        client.post("/file/is-processed", json={"filename": filename}).get_json()[
            "is_processed"
        ]
        is True
    )


def test_chat_is_blocked_while_a_ready_document_is_reindexed(client, app):
    """A reindex never exposes an old or partial generation to chat."""
    filename = index_document(client, app)
    assert client.post(f"/ingestion-jobs/{filename}/retry").status_code == 201

    response = client.post("/response", json={"query": "what?", "filename": filename})

    assert response.status_code == 409
    assert response.get_json()["category"] == "document_indexing"


def test_ask_streams_tokens_and_persists_sources(client, app, fake_vectors):
    """Do test ask streams tokens and persists sources."""
    filename = index_document(client, app)

    response = client.post("/response", json={"query": "what?", "filename": filename})

    assert response.status_code == 200
    assert response.mimetype == "text/event-stream"
    events = parse_sse(response.get_data(as_text=True))

    tokens = [data["text"] for name, data in events if name == "token"]
    assert "".join(tokens) == "The answer is 42.", "tokens should arrive as fragments"

    done_name, done_data = events[-1]
    assert done_name == "done"
    assert done_data["done"] is True
    assert done_data["retrieval"] == {
        "method": "dense",
        "outcome": "success",
        "original_query": "what?",
        "expanded_query": "what?",
        "query_expansion": "none",
        "dropped_turns": 0,
        "dropped_sources": 0,
    }
    assert len(done_data["sources"]) == 1
    src = done_data["sources"][0]
    assert src["content"] == "chunk about topic"
    assert src["document"] == "doc.pdf"
    assert src["chunk_index"] == 0
    assert src["score"] == 0.92
    assert "page" in src

    history = client.get(f"/messages?filename={filename}").get_json()["messages"]
    assert [(m["sender"], m["text"]) for m in history] == [
        ("user", "what?"),
        ("bot", "The answer is 42."),
    ]
    assert history[1]["sources"] == done_data["sources"], (
        "the terminal event must carry the sources persisted with the answer"
    )


def test_generation_failure_does_not_print_the_api_key(capsys):
    """One-shot generation diagnostics cannot expose the provider key."""
    secret = "sk-generation-secret"

    class FailingProvider(LLMProvider):
        def _build_client(self):
            return None

        def verify(self):
            return None

        def _generate_response(self, query, context):
            raise RuntimeError(f"provider echoed {secret}")

        def _stream_response(self, query, context):
            yield ""

    provider = FailingProvider(secret, "test-model")
    result = provider.generate_response("question", "retrieved passage")

    assert secret not in capsys.readouterr().out
    assert "retrieved passage" in result


def test_streaming_provider_failure_does_not_log_the_api_key(
    client, app, fake_vectors, fake_chat, caplog
):
    """Streaming diagnostics cannot expose the stored provider key."""
    filename = index_document(client, app)
    fake_chat.provider_error = RuntimeError("provider echoed sk-test-chat-key")

    with caplog.at_level(logging.ERROR):
        response = client.post(
            "/response", json={"query": "what?", "filename": filename}
        )

    assert response.status_code == 200
    assert "sk-test-chat-key" not in caplog.text


def test_factory_builds_provider_with_credentials_arguments():
    """Provider/model/key land on the instance the factory builds."""
    from services.llm.factory import build_chat_provider
    from services.llm.google_provider import GoogleProvider
    from services.llm.groq_provider import GroqProvider

    groq = build_chat_provider(
        ChatCredentials(provider="groq", model="m1", api_key="key-a")
    )
    assert isinstance(groq, GroqProvider)
    assert (groq.api_key, groq.model) == ("key-a", "m1")

    google = build_chat_provider(
        ChatCredentials(provider="google", model="m2", api_key="key-b")
    )
    assert isinstance(google, GoogleProvider)
    assert (google.api_key, google.model) == ("key-b", "m2")


def test_factory_rejects_unknown_provider():
    """An unknown provider name is a hard error, not a silent default."""
    from services.llm.factory import build_chat_provider

    credentials = ChatCredentials(provider="anthropic", model="m1", api_key="key-a")
    with pytest.raises(ValueError, match="Unsupported provider"):
        build_chat_provider(credentials)


def test_provider_failure_surfaces_without_fallback():
    """A primary failure propagates; the factory wires no fallback provider."""
    from services.llm.factory import build_chat_provider

    class _ProviderDown:
        """Any SDK access fails: the provider is unreachable."""

        def __init__(self):
            self._message = "provider down"

        def __getattr__(self, name):
            raise RuntimeError(self._message)

    provider = build_chat_provider(
        ChatCredentials(provider="groq", model="m1", api_key="key-a"),
        client=_ProviderDown(),
    )
    with pytest.raises(RuntimeError, match="provider down"):
        list(provider.stream_response("q", "ctx"))


def test_response_rejects_a_retired_saved_model(client, app, repositories):
    """A stored model outside the active catalog cannot serve new chats."""
    filename = index_document(client, app)
    repositories.app_settings.upsert_app_settings(
        "google", "gemini-2.0-flash", encrypt_api_key("sk-retired-model-key")
    )

    response = client.post("/response", json={"query": "what?", "filename": filename})

    assert response.status_code == 400
    assert "unsupported model" in response.get_json()["error"].lower()


def test_response_rejects_an_unsupported_saved_provider(client, app, repositories):
    """A stored provider outside the catalog gets an actionable error."""
    filename = index_document(client, app)
    repositories.app_settings.upsert_app_settings(
        "anthropic", "claude-test", encrypt_api_key("sk-unsupported-provider-key")
    )

    response = client.post("/response", json={"query": "what?", "filename": filename})

    assert response.status_code == 400
    assert "unsupported provider" in response.get_json()["error"].lower()


def test_response_rejects_malformed_saved_settings(client, app, repositories):
    """An incomplete settings row gets re-save guidance, not a crash."""
    filename = index_document(client, app)
    repositories.app_settings.upsert_app_settings("", "", "not-a-key")

    response = client.post("/response", json={"query": "what?", "filename": filename})

    assert response.status_code == 400
    assert "incomplete" in response.get_json()["error"]


def test_chat_without_settings_asks_user_to_configure(client, repositories):
    """No saved global settings yields a settings-oriented 400, not an LLM call."""
    with repositories.app_settings._session_factory() as session, session.begin():
        session.query(AppSettings).delete(synchronize_session=False)

    filename = upload(client).get_json()["file"]["name"]
    response = client.post("/response", json={"query": "what?", "filename": filename})

    assert response.status_code == 400
    assert "Settings" in response.get_json()["error"]


def test_provider_failure_still_leaves_a_visible_reply(
    client, app, fake_vectors, fake_chat, capsys
):
    """Do test provider failure still leaves a visible reply."""
    filename = index_document(client, app)

    fake_chat.provider_error = RuntimeError("provider down")

    response = client.post("/response", json={"query": "what?", "filename": filename})

    events = parse_sse(response.get_data(as_text=True))
    names = [name for name, _ in events]
    assert "error" in names, "a provider failure must surface as an error event"

    error_data = next(data for name, data in events if name == "error")
    assert error_data["error"], "the error event should carry a readable message"

    history = client.get(f"/messages?filename={filename}").get_json()["messages"]
    assert [m["sender"] for m in history] == ["user", "bot"], (
        "a failed turn must still end with an assistant reply, not a stranded question"
    )
    assert error_data["error"] in history[-1]["text"]


def test_sources_panel_order_matches_llm_context_order(
    client, app, fake_vectors, fake_chat
):
    """Do test sources panel order matches llm context order."""
    filename = index_document(client, app)

    fake_vectors.matches = [
        {
            "content": "weak chunk",
            "document": "doc.pdf",
            "chunk_index": 2,
            "score": 0.31,
        },
        {
            "content": "strong chunk",
            "document": "doc.pdf",
            "chunk_index": 1,
            "score": 0.95,
        },
    ]

    response = client.post("/response", json={"query": "what?", "filename": filename})

    done_sources = parse_sse(response.get_data(as_text=True))[-1][1]["sources"]
    assert [s["content"] for s in done_sources] == ["strong chunk", "weak chunk"], (
        "sources must be ordered by score, not by index order"
    )

    # The LLM received the same chunks, in the same order, as its context.
    _, context, _ = fake_chat.streamed[0]
    assert context == "strong chunk\n\nweak chunk"

    history = client.get(f"/messages?filename={filename}").get_json()["messages"]
    assert history[1]["sources"] == done_sources, (
        "the persisted sources must replay in the same order for history"
    )


def test_delete_removes_everything_with_no_orphans(
    client, app, fake_storage, fake_vectors, repositories
):
    """Do test delete removes everything with no orphans."""
    filename = index_document(client, app)
    client.post("/response", json={"query": "what?", "filename": filename})

    response = client.delete(f"/files/remove?path={filename}")

    assert response.status_code == 200
    assert fake_storage.blobs == {}
    # Process-file now cleans stale vectors before upsert, so the filename
    # appears once from that cleanup plus once from the explicit delete.
    assert filename in fake_vectors.deleted
    assert fake_vectors.deleted.count(filename) >= 1
    assert repositories.files.list_files() == []
    assert client.get(f"/messages?filename={filename}").get_json()["messages"] == []
    assert client.get("/files").get_json()["files"] == []


def test_process_embed_failure_compensates_with_no_orphans(
    client, app, fake_vectors, fake_embeddings
):
    """An embed failure removes this doc's vectors and leaves it not processed."""
    filename = upload(client).get_json()["file"]["name"]

    def _boom(texts):
        raise RuntimeError("embed down")

    fake_embeddings.embed_texts = _boom

    response = client.post("/process-file", json={"filename": filename})

    assert response.status_code == 202
    assert fake_vectors.upserts == []
    assert run_ingestion(app) == 1

    job = app.config["TEST_REPOSITORIES"].ingestion_jobs.get_latest(filename)
    assert job["state"] == "failed"
    assert job["error_category"] == "ingestion_failed"
    assert fake_vectors.deleted.count(filename) >= 1
    assert (
        client.post("/file/is-processed", json={"filename": filename}).get_json()[
            "is_processed"
        ]
        is False
    )


def test_process_upsert_failure_compensates_with_no_orphans(client, app, fake_vectors):
    """An upsert failure removes this doc's vectors and leaves it not processed."""
    filename = upload(client).get_json()["file"]["name"]

    def _boom(embeddings, chunks, fname, **kwargs):
        raise RuntimeError("upsert down")

    fake_vectors.upsert_chunks = _boom

    response = client.post("/process-file", json={"filename": filename})

    assert response.status_code == 202
    assert run_ingestion(app) == 1
    assert fake_vectors.deleted.count(filename) >= 1
    assert (
        client.post("/file/is-processed", json={"filename": filename}).get_json()[
            "is_processed"
        ]
        is False
    )


@pytest.mark.parametrize(
    ("error", "expected_category"),
    [
        (
            VectorStoreUnavailableError("Vector store is unavailable"),
            "vector_store_unavailable",
        ),
        (
            VectorStoreConfigurationError("Collection schema is invalid"),
            "vector_store_configuration",
        ),
        (
            VectorDimensionError(expected=1024, got=512),
            "vector_dimension_mismatch",
        ),
    ],
)
def test_ingestion_job_reports_vector_failure_categories(
    client, app, fake_vectors, error, expected_category
):
    """Vector-store failures stay distinct on the durable job."""

    def _fail(*args, **kwargs):
        raise error

    filename = upload(client).get_json()["file"]["name"]
    fake_vectors.upsert_chunks = _fail

    assert client.post("/process-file", json={"filename": filename}).status_code == 202
    assert run_ingestion(app) == 1

    job = app.config["TEST_REPOSITORIES"].ingestion_jobs.get_latest(filename)
    assert job["state"] == "failed"
    assert job["error_category"] == expected_category
    assert (
        client.post("/file/is-processed", json={"filename": filename}).get_json()[
            "is_processed"
        ]
        is False
    )


def test_process_retry_cleans_stale_vectors_before_rewrite(client, app, fake_vectors):
    """A retry after failure cleans stale vectors first, so it stays idempotent."""
    filename = upload(client).get_json()["file"]["name"]

    def _boom(embeddings, chunks, fname, **kwargs):
        raise RuntimeError("upsert down")

    fake_vectors.upsert_chunks = _boom
    assert client.post("/process-file", json={"filename": filename}).status_code == 202
    assert run_ingestion(app) == 1
    assert fake_vectors.deleted.count(filename) >= 1

    # Fix the index and retry with a working upsert.
    upserts = []
    deletes = fake_vectors.deleted

    def _ok(embeddings, chunks, fname, **kwargs):
        upserts.append((embeddings, chunks, fname))

    fake_vectors.upsert_chunks = _ok
    retry = client.post(f"/ingestion-jobs/{filename}/retry")
    assert retry.status_code == 201
    assert run_ingestion(app) == 1
    # The retry cleaned stale vectors before writing again.
    assert deletes.count(filename) >= 2
    assert upserts and upserts[0][2] == filename
    assert (
        client.post("/file/is-processed", json={"filename": filename}).get_json()[
            "is_processed"
        ]
        is True
    )


def test_empty_retrieval_is_a_successful_empty_result(client, app, fake_vectors):
    """A valid query with no evidence completes with an empty result marker."""
    filename = index_document(client, app)
    fake_vectors.matches = []

    response = client.post("/response", json={"query": "what?", "filename": filename})

    assert response.status_code == 200
    done = parse_sse(response.get_data(as_text=True))[-1][1]
    assert done["sources"] == []
    assert done["retrieval"] == {
        "method": "dense",
        "outcome": "empty",
        "original_query": "what?",
        "expanded_query": "what?",
        "query_expansion": "none",
        "dropped_turns": 0,
        "dropped_sources": 0,
    }


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_category"),
    [
        (
            VectorStoreUnavailableError("Vector store is unavailable"),
            503,
            "vector_store_unavailable",
        ),
        (
            VectorStoreConfigurationError("Collection schema is invalid"),
            500,
            "vector_store_configuration",
        ),
        (
            VectorDimensionError(expected=1024, got=512),
            409,
            "vector_dimension_mismatch",
        ),
    ],
)
def test_retrieval_failure_categories_are_distinct_at_http_boundary(
    client, app, fake_vectors, error, expected_status, expected_category
):
    """Each retrieval failure category has its own HTTP status and payload."""
    filename = index_document(client, app)

    def _fail(*args, **kwargs):
        raise error

    fake_vectors.query_vectors = _fail
    response = client.post("/response", json={"query": "what?", "filename": filename})

    assert response.status_code == expected_status
    assert response.get_json()["category"] == expected_category
    assert client.get(f"/messages?filename={filename}").get_json()["messages"] == []


def test_chat_retrieval_failure_leaves_no_stranded_message(client, app, fake_vectors):
    """A retrieval failure persists no user message, so no stranded question."""
    filename = index_document(client, app)

    def _boom(embedding, fname, **kwargs):
        raise RuntimeError("index down")

    fake_vectors.query_vectors = _boom

    response = client.post("/response", json={"query": "what?", "filename": filename})

    assert response.status_code == 500
    history = client.get(f"/messages?filename={filename}").get_json()["messages"]
    assert history == []


def test_chat_embed_failure_leaves_no_stranded_message(client, app, fake_embeddings):
    """An embed failure on the query path also leaves no stranded question."""
    filename = index_document(client, app)

    def _boom(texts):
        raise RuntimeError("embed down")

    fake_embeddings.embed_texts = _boom

    response = client.post("/response", json={"query": "what?", "filename": filename})

    assert response.status_code == 500
    history = client.get(f"/messages?filename={filename}").get_json()["messages"]
    assert history == []


def test_chat_query_too_long_rejected_before_embedding(client, app, fake_embeddings):
    """An oversized query is a 400 and never reaches the embedding model."""
    from routes.chat import MAX_QUERY_CHARS

    filename = index_document(client, app)
    embedded_before = list(fake_embeddings.embedded)

    response = client.post(
        "/response", json={"query": "x" * (MAX_QUERY_CHARS + 1), "filename": filename}
    )

    assert response.status_code == 400
    assert "too long" in response.get_json()["error"].lower()
    assert fake_embeddings.embedded == embedded_before
    history = client.get(f"/messages?filename={filename}").get_json()["messages"]
    assert history == []


def test_conversation_repository_round_trip():
    """Turn, status, and source records survive their aggregate interface."""
    from repositories import ConversationRepository, FileRepository

    engine = create_engine(
        "sqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    file_record = FileRepository(session_factory).create_file("doc.pdf", title="Doc")
    repository = ConversationRepository(session_factory)

    conversation_id = repository.ensure_conversation(file_record["id"])
    turn_id = repository.start_turn(conversation_id, "Question")
    repository.complete_turn(
        turn_id,
        "Answer",
        [
            {
                "content": "Source",
                "document": "doc.pdf",
                "chunk_index": 0,
                "score": 0.9,
                "page": 3,
            }
        ],
    )

    turns = repository.get_turns(conversation_id)
    assert [turn["status"] for turn in turns] == ["answered"]
    assert [turn["answer"] for turn in turns] == ["Answer"]
    assert turns[0]["sources"] == [
        {
            "content": "Source",
            "document": "doc.pdf",
            "chunk_index": 0,
            "score": 0.9,
            "page": 3,
        }
    ]
    assert [
        message["text"] for message in repository.get_messages(conversation_id)
    ] == [
        "Question",
        "Answer",
    ]


def test_concurrent_batches_cancel_on_first_business_failure():
    """Remaining batches cancel on the first business error instead of running on."""
    import threading
    import time

    from services.concurrency import map_batches_concurrently

    started = []
    started_lock = threading.Lock()

    def _func(batch):
        with started_lock:
            started.append(batch)
        if batch == "bad":
            raise RuntimeError("business boom")
        time.sleep(0.05)
        return batch

    batches = ["bad"] + [f"ok-{i}" for i in range(9)]
    with pytest.raises(RuntimeError, match="business boom"):
        map_batches_concurrently(
            batches,
            _func,
            label="test-cancel",
            max_workers=1,
        )
    # The bad batch fails fast; queued batches are cancelled so the full
    # tail never runs (without cancel all 10 would start).
    assert "bad" in started
    assert len(started) < len(batches)


def test_fresh_database_reaches_current_schema_via_migrations(tmp_path):
    """Running the migrations on an empty database produces the current schema."""
    backend_dir = pathlib.Path(__file__).resolve().parent.parent
    app_packages = {
        "app",
        "composition",
        "db",
        "providers",
        "repositories",
        "routes",
        "services",
        "settings",
        "storage",
    }
    app_imports = []
    for migration in (backend_dir / "migrations").rglob("*.py"):
        tree = ast.parse(migration.read_text(encoding="utf-8"), filename=str(migration))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                if module.split(".", 1)[0] in app_packages:
                    app_imports.append(f"{migration.name}:{module}")

    assert app_imports == []

    db_path = tmp_path / "fresh.db"
    env = {
        **{k: v for k, v in os.environ.items() if k != "DATABASE_URL"},
        "DATABASE_URL": f"sqlite:///{db_path}",
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(pathlib.Path.home()),
    }
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        capture_output=True,
        text=True,
        cwd=backend_dir,
        env=env,
    )
    assert result.returncode == 0, result.stderr

    import sqlite3

    connection = sqlite3.connect(db_path)
    tables = {
        row[0]
        for row in connection.execute(
            "select name from sqlite_master where type='table'"
        )
    }
    assert {
        "alembic_version",
        "app_settings",
        "conversations",
        "files",
        "index_generation_cleanups",
        "ingestion_jobs",
        "sources",
        "turns",
    } <= tables
    assert {"messages", "users", "user_settings"}.isdisjoint(tables)
    assert connection.execute("select version_num from alembic_version").fetchone() == (
        "d4f1a8b3c6e2",
    )
    assert {
        "title",
        "original_filename",
        "is_processed",
        "last_opened_at",
        "index_generation",
    } <= {row[1] for row in connection.execute("pragma table_info(files)")}
    assert {"index_manifest", "index_stale_reason"} <= {
        row[1] for row in connection.execute("pragma table_info(files)")
    }
    assert {
        "cancel_requested_at",
        "cancelled_at",
        "limits_json",
        "usage_json",
    } <= {row[1] for row in connection.execute("pragma table_info(ingestion_jobs)")}
    assert {"deletion_state", "deletion_error", "deletion_attempts"} <= {
        row[1] for row in connection.execute("pragma table_info(files)")
    }
    assert {"page", "turn_id"} <= {
        row[1] for row in connection.execute("pragma table_info(sources)")
    }
    assert {
        "sequence",
        "question",
        "answer",
        "status",
        "failure_reason",
        "created_at",
        "completed_at",
    } <= {row[1] for row in connection.execute("pragma table_info(turns)")}
    connection.close()
