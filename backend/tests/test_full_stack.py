"""Full-stack tests over HTTP with real infrastructure and deterministic models."""

import os
import pathlib
import subprocess
import sys
import uuid
from dataclasses import dataclass

import psycopg
import pytest
from psycopg import sql
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from db import Conversation, Message, Source
from services.embeddings.local_embeddings import EmbeddingService
from services.llm.base import LLMProvider
from services.retrieval.base import VectorStore
from services.retrieval.reranker import Reranker
from settings import (
    AuthSettings,
    ChunkingSettings,
    DatabaseSettings,
    EmbeddingSettings,
    FrontendSettings,
    ParsingSettings,
    RerankSettings,
    Settings,
    StorageSettings,
    UploadSettings,
    VectorSettings,
)
from storage import LocalStorage
from tests.fullstack_support import ApplicationHarness, build_application
from tests.sse import parse_sse

BACKEND_DIR = pathlib.Path(__file__).resolve().parent.parent
SAMPLE_PDF = BACKEND_DIR / "evaluation" / "sample_docs" / "papermind-rag-primer.pdf"
ADMIN_DATABASE_URL = os.getenv(
    "FULL_STACK_DATABASE_URL",
    "postgresql+psycopg://papermind:papermind@127.0.0.1:55432/papermind",
)
QDRANT_URL = os.getenv("FULL_STACK_QDRANT_URL", "http://127.0.0.1:56333")


@dataclass(frozen=True)
class MigratedDatabase:
    """Connection details for one freshly migrated database."""

    url: str
    engine: object


def _psycopg_url(database_url: str) -> str:
    """Convert a SQLAlchemy PostgreSQL URL to a psycopg connection URL."""
    url = make_url(database_url)
    return url.set(drivername=url.drivername.replace("+psycopg", "")).render_as_string(
        hide_password=False
    )


def _drop_database(database_name: str) -> None:
    """Terminate open connections and remove an isolated test database."""
    with psycopg.connect(
        _psycopg_url(ADMIN_DATABASE_URL), autocommit=True
    ) as connection:
        connection.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE datname = %s AND pid <> pg_backend_pid()",
            (database_name,),
        )
        connection.execute(
            sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(database_name))
        )


@pytest.fixture
def migrated_database():
    """Create, migrate, and remove one isolated Postgres database."""
    if os.getenv("RUN_FULL_STACK_TESTS") != "1":
        pytest.skip("set RUN_FULL_STACK_TESTS=1 to run full-stack tests")

    database_name = f"papermind_test_{uuid.uuid4().hex}"
    database_url = (
        make_url(ADMIN_DATABASE_URL)
        .set(database=database_name)
        .render_as_string(hide_password=False)
    )
    with psycopg.connect(
        _psycopg_url(ADMIN_DATABASE_URL), autocommit=True
    ) as connection:
        connection.execute(
            sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name))
        )

    try:
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=BACKEND_DIR,
            env={**os.environ, "DATABASE_URL": database_url},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        engine = create_engine(database_url, pool_pre_ping=True)
        try:
            yield MigratedDatabase(database_url, engine)
        finally:
            engine.dispose()
    finally:
        _drop_database(database_name)


@pytest.fixture
def full_stack_app(migrated_database, tmp_path):
    """Serve the app over HTTP with isolated production dependencies."""
    collection_name = f"full-stack-{uuid.uuid4().hex}"
    app_settings = Settings(
        database=DatabaseSettings(database_url=migrated_database.url),
        storage=StorageSettings(storage_dir=tmp_path / "storage"),
        vector=VectorSettings(
            qdrant_url=QDRANT_URL,
            index_name=collection_name,
        ),
        embedding=EmbeddingSettings(embedding_model="deterministic"),
        chunking=ChunkingSettings(
            chunk_size_tokens=512,
            chunk_overlap_tokens=50,
        ),
        rerank=RerankSettings(
            rerank_model="deterministic",
            enabled=True,
        ),
        parsing=ParsingSettings(use_docling="false"),
        auth=AuthSettings(app_secret="full-stack-test-secret"),
        upload=UploadSettings(),
        frontend=FrontendSettings(frontend_origin="http://localhost:5173"),
    )
    session_factory = sessionmaker(
        bind=migrated_database.engine,
        expire_on_commit=False,
    )
    harness = build_application(app_settings, session_factory)
    try:
        yield harness
    finally:
        harness.close()


def _configure_chat(harness: ApplicationHarness) -> None:
    """Save and verify credentials through the public settings routes."""
    response = harness.client.put(
        "/settings",
        json={
            "provider": "groq",
            "model": "llama-3.3-70b-versatile",
            "api_key": "full-stack-test-key",
        },
    )
    assert response.status_code == 200, response.text
    response = harness.client.post("/settings/verify")
    assert response.status_code == 200, response.text
    assert response.json() == {"ok": True, "error": None}


def _upload_sample(harness: ApplicationHarness) -> str:
    """Upload the representative PDF through the HTTP route."""
    with SAMPLE_PDF.open("rb") as pdf:
        response = harness.client.post(
            "/upload",
            files={"file": (SAMPLE_PDF.name, pdf, "application/pdf")},
        )
    assert response.status_code == 200, response.text
    return response.json()["file"]["name"]


def _process(harness: ApplicationHarness, filename: str) -> None:
    """Process an uploaded PDF through the HTTP route."""
    response = harness.client.post("/process-file", json={"filename": filename})
    assert response.status_code == 200, response.text


@pytest.mark.full_stack
def test_migrations_apply_to_an_empty_postgres_database(migrated_database):
    """A new database reaches the current schema without residual rows."""
    engine = migrated_database.engine

    assert set(inspect(engine).get_table_names()) == {
        "alembic_version",
        "app_settings",
        "conversations",
        "files",
        "messages",
        "sources",
    }
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT version_num FROM alembic_version")) == (
            "c4d2e9a1b5f6"
        )
        for table in ("app_settings", "conversations", "files", "messages", "sources"):
            assert connection.scalar(text(f"SELECT count(*) FROM {table}")) == 0


@pytest.mark.full_stack
def test_pdf_flow_persists_across_http_postgres_qdrant_and_storage(
    full_stack_app,
):
    """A PDF can be uploaded, processed, streamed, reloaded, and deleted."""
    harness = full_stack_app

    assert harness.client.get("/health").json() == {"response": "OK"}
    _configure_chat(harness)
    filename = _upload_sample(harness)

    unprocessed = harness.client.post(
        "/file/is-processed",
        json={"filename": filename},
    )
    assert unprocessed.status_code == 200
    assert unprocessed.json() == {"is_processed": False}

    _process(harness, filename)
    assert harness.repositories.files.get_file(filename)["is_processed"] is True
    points, _ = harness.qdrant.scroll(
        collection_name=harness.collection_name,
        limit=100,
        with_payload=True,
        with_vectors=True,
    )
    assert points
    assert any(
        "A RAG pipeline has five stages" in (point.payload or {}).get("content", "")
        for point in points
    )
    assert harness.qdrant.count(
        collection_name=harness.collection_name,
        exact=True,
    ).count == len(points)

    metadata = harness.client.get(f"/files/{filename}/meta")
    assert metadata.status_code == 200
    assert metadata.json()["pageCount"] == 2
    download = harness.client.get(f"/storage/{filename}")
    assert download.status_code == 200
    assert download.content == SAMPLE_PDF.read_bytes()

    response = harness.client.post(
        "/response",
        json={
            "filename": filename,
            "query": "What are the five stages of a RAG pipeline?",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    tokens = [payload["text"] for name, payload in events if name == "token"]
    assert "".join(tokens) == "The answer is grounded in the retrieved PDF text."
    terminal = events[-1]
    assert terminal[0] == "done"
    assert terminal[1]["done"] is True
    assert any(
        "A RAG pipeline has five stages" in source["content"]
        for source in terminal[1]["sources"]
    )
    assert "A RAG pipeline has five stages" in harness.chat.streamed[-1][1]

    history = harness.client.get(f"/messages?filename={filename}")
    assert history.status_code == 200
    messages = history.json()["messages"]
    assert [message["sender"] for message in messages] == ["user", "bot"]
    assert messages[1]["text"] == "The answer is grounded in the retrieved PDF text."
    source_fields = ("content", "document", "chunk_index", "score", "page")
    assert [
        {field: source[field] for field in source_fields}
        for source in messages[1]["sources"]
    ] == [
        {field: source[field] for field in source_fields}
        for source in terminal[1]["sources"]
    ]

    with harness.session_factory() as session:
        assert session.query(Message).count() == 2
        assert session.query(Source).count() == len(terminal[1]["sources"])

    assert isinstance(harness.storage, LocalStorage)
    assert isinstance(harness.embeddings, EmbeddingService)
    assert isinstance(harness.reranker, Reranker)
    assert isinstance(harness.vector_store, VectorStore)
    assert all(isinstance(provider, LLMProvider) for provider in harness.chat.built)

    deleted = harness.client.delete(f"/files/remove?path={filename}")
    assert deleted.status_code == 200
    assert harness.storage.list() == []
    assert harness.repositories.files.get_file(filename) is None
    assert (
        harness.qdrant.count(
            collection_name=harness.collection_name,
            exact=True,
        ).count
        == 0
    )
    with harness.session_factory() as session:
        assert session.query(Conversation).count() == 0
        assert session.query(Message).count() == 0
        assert session.query(Source).count() == 0


@pytest.mark.full_stack
@pytest.mark.parametrize(
    ("boundary", "expected_status", "expected_event"),
    [
        pytest.param("storage", 500, None, id="storage"),
        pytest.param("database", 500, None, id="database"),
        pytest.param("vector", 500, None, id="vector"),
        pytest.param("embedding", 500, None, id="embedding"),
        pytest.param("reranking", 500, None, id="reranking"),
        pytest.param("provider", 200, "error", id="provider"),
    ],
)
def test_injected_service_failures_reach_visible_http_responses(
    full_stack_app,
    boundary,
    expected_status,
    expected_event,
):
    """Each external dependency can fail without bypassing the HTTP route."""
    harness = full_stack_app
    _configure_chat(harness)

    if boundary == "storage":
        harness.storage.fail("save")
        with SAMPLE_PDF.open("rb") as pdf:
            response = harness.client.post(
                "/upload",
                files={"file": (SAMPLE_PDF.name, pdf, "application/pdf")},
            )
        assert response.status_code == expected_status
        assert response.json() == {"error": "Internal server error"}
        assert harness.storage.list() == []
        assert harness.repositories.files.list_files() == []
        return

    filename = _upload_sample(harness)
    if boundary == "database":
        harness.repositories.files.fail("get_file")
    else:
        _process(harness, filename)
        if boundary == "vector":
            harness.vector_store.fail("query")
        elif boundary == "embedding":
            harness.embeddings.fail("embed")
        elif boundary == "reranking":
            harness.reranker.fail("rerank")
        elif boundary == "provider":
            harness.chat.fail("stream")

    response = harness.client.post(
        "/response",
        json={"filename": filename, "query": "How does RAG retrieve evidence?"},
    )

    assert response.status_code == expected_status
    if expected_event is None:
        assert response.json() == {"error": "Internal server error"}
    else:
        events = parse_sse(response.text)
        assert [name for name, _ in events] == ["error", "done"]
        assert events[0][1]["error"]
        assert events[1][1] == {"done": True, "sources": []}
