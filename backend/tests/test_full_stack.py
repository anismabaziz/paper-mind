"""Full-stack tests over HTTP with real infrastructure and deterministic models."""

import json
import os
import pathlib
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass

import psycopg
import pytest
from qdrant_client.models import FieldCondition, Filter as QdrantFilter, MatchValue
from psycopg import sql
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from db import Conversation, FileRecord, IngestionJob, Source, Turn
from services.embeddings.local_embeddings import EmbeddingService
from services.llm.base import LLMProvider
from services.retrieval.base import VectorStore, VectorStoreConfigurationError
from services.retrieval.hybrid import SPARSE_METHOD, TOKENIZER_VERSION
from services.retrieval.qdrant_store import QdrantIndexAdapter
from services.retrieval.reranker import Reranker
from services.retrieval.vector_service import VectorService
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
PRE_TURN_REVISION = "c3d7e1f4a9b2"
CURRENT_REVISION = "d4f1a8b3c6e2"


@dataclass(frozen=True)
class MigratedDatabase:
    """Connection details for one freshly migrated database."""

    url: str
    engine: object


def _create_database(target_revision: str):
    """Create an empty database, migrate it to a revision, and yield its details."""
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
            [sys.executable, "-m", "alembic", "upgrade", target_revision],
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
    yield from _create_database("head")


@pytest.fixture
def legacy_database():
    """Create a database migrated to the last revision before ordered turns."""
    if os.getenv("RUN_FULL_STACK_TESTS") != "1":
        pytest.skip("set RUN_FULL_STACK_TESTS=1 to run full-stack tests")
    yield from _create_database(PRE_TURN_REVISION)


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
    candidate = {
        "provider": "groq",
        "model": "openai/gpt-oss-20b",
        "api_key": "full-stack-test-key",
    }
    response = harness.client.put("/settings", json=candidate)
    assert response.status_code == 200, response.text
    response = harness.client.post("/settings/verify", json=candidate)
    assert response.status_code == 200, response.text
    assert response.json()["ok"] is True
    assert response.json()["error"] is None


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
    """Queue indexing for an uploaded PDF, then run the worker to ready."""
    response = harness.client.post("/process-file", json={"filename": filename})
    assert response.status_code == 202, response.text
    assert harness.drain() == 1


def _claim_in_parallel(jobs, *worker_ids: str) -> list[dict | None]:
    """Race several workers claiming the queue at the same time."""
    barrier = threading.Barrier(len(worker_ids))
    results: list[dict | None] = []
    lock = threading.Lock()

    def claim(worker_id: str) -> None:
        barrier.wait()
        try:
            job = jobs.claim_next(worker_id)
        except Exception:
            job = None
        with lock:
            results.append(job)

    threads = [
        threading.Thread(target=claim, args=(worker_id,)) for worker_id in worker_ids
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    return results


@pytest.mark.full_stack
def test_failed_candidate_keeps_the_saved_provider_credentials(full_stack_app):
    """A rejected candidate leaves the real settings row usable."""
    harness = full_stack_app
    working = {
        "provider": "groq",
        "model": "openai/gpt-oss-20b",
        "api_key": "working-key",
    }
    assert harness.client.put("/settings", json=working).status_code == 200
    before = harness.repositories.app_settings.get_app_settings()
    harness.chat.fail("verify")

    response = harness.client.put(
        "/settings",
        json={
            "provider": "google",
            "model": "gemini-2.5-flash",
            "api_key": "bad-key",
        },
    )

    assert response.status_code == 400
    assert "not changed" in response.json()["error"]
    after = harness.repositories.app_settings.get_app_settings()
    assert after["provider"] == before["provider"]
    assert after["model"] == before["model"]
    assert after["encrypted_api_key"] == before["encrypted_api_key"]
    assert harness.client.get("/settings").json()["masked_key"] == "••••-key"


@pytest.mark.full_stack
def test_migrations_apply_to_an_empty_postgres_database(migrated_database):
    """A new database reaches the current schema without residual rows."""
    engine = migrated_database.engine

    assert set(inspect(engine).get_table_names()) == {
        "alembic_version",
        "app_settings",
        "conversations",
        "files",
        "index_generation_cleanups",
        "ingestion_jobs",
        "sources",
        "turns",
    }
    with engine.connect() as connection:
        assert (
            connection.scalar(text("SELECT version_num FROM alembic_version"))
            == CURRENT_REVISION
        )
        for table in (
            "app_settings",
            "conversations",
            "files",
            "ingestion_jobs",
            "sources",
            "turns",
        ):
            assert connection.scalar(text(f"SELECT count(*) FROM {table}")) == 0


@pytest.mark.full_stack
def test_qdrant_vectors_and_all_retrieval_methods_survive_adapter_restart(
    full_stack_app,
):
    """Stored named vectors answer every mode through a fresh adapter."""
    harness = full_stack_app
    filename = _upload_sample(harness)
    _process(harness, filename)

    collection = harness.qdrant.get_collection(harness.collection_name)
    dense_schema = collection.config.params.vectors
    sparse_schema = collection.config.params.sparse_vectors
    assert set(dense_schema) == {"dense"}
    assert set(sparse_schema) == {"sparse"}
    assert dense_schema["dense"].size == 1024

    points, _ = harness.qdrant.scroll(
        collection_name=harness.collection_name,
        limit=100,
        with_payload=True,
        with_vectors=True,
    )
    assert points
    assert all(set(point.vector) == {"dense", "sparse"} for point in points)
    assert all(point.vector["dense"] for point in points)
    assert all(point.vector["sparse"].indices for point in points)
    assert all(point.payload["sparse_method"] == SPARSE_METHOD for point in points)
    assert all(
        point.payload["sparse_tokenizer_version"] == TOKENIZER_VERSION
        for point in points
    )

    restarted_store = QdrantIndexAdapter(harness.qdrant, harness.collection_name)
    service = VectorService(restarted_store)
    query_text = "What are the five stages of a RAG pipeline?"
    embedding = harness.embeddings.embed_texts(query_text)[0]

    for method in ("dense", "sparse", "hybrid"):
        result = service.query_vectors(
            embedding,
            filename,
            query_text=query_text,
            method=method,
        )
        assert result.method == method
        assert result.outcome == "success"
        assert any(
            "A RAG pipeline has five stages" in source["content"]
            for source in result.sources
        )


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
    assert unprocessed.json()["is_processed"] is False
    assert unprocessed.json()["ingestion"]["state"] == "queued"

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
    assert terminal[1]["retrieval"] == {"method": "hybrid", "outcome": "success"}
    assert any(
        "A RAG pipeline has five stages" in source["content"]
        for source in terminal[1]["sources"]
    )
    assert "A RAG pipeline has five stages" in harness.chat.streamed[-1][1]
    assert any("rerank_score" in source for source in terminal[1]["sources"])

    history = harness.client.get(f"/messages?filename={filename}")
    assert history.status_code == 200
    messages = history.json()["messages"]
    assert [message["sender"] for message in messages] == ["user", "bot"]
    assert messages[1]["text"] == "The answer is grounded in the retrieved PDF text."
    assert [message["turn_sequence"] for message in messages] == [1, 1]
    assert [message["turn_status"] for message in messages] == ["answered", "answered"]
    source_fields = ("content", "document", "chunk_index", "score", "page")
    assert [
        {field: source[field] for field in source_fields}
        for source in messages[1]["sources"]
    ] == [
        {field: source[field] for field in source_fields}
        for source in terminal[1]["sources"]
    ]

    with harness.session_factory() as session:
        assert session.query(Turn).count() == 1
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
        assert session.query(Turn).count() == 0
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
        assert events[1][1] == {
            "done": True,
            "sources": [],
            "retrieval": {"method": "hybrid", "outcome": "success"},
        }


@pytest.mark.full_stack
@pytest.mark.parametrize(
    "boundary",
    [
        pytest.param("disk", id="disk"),
        pytest.param("postgres", id="postgres"),
        pytest.param("qdrant", id="qdrant"),
        pytest.param("partial", id="partial"),
    ],
)
def test_deletion_failures_retain_retryable_state(full_stack_app, boundary):
    """Disk, Postgres, Qdrant, and partial cleanup failures stay retryable."""
    harness = full_stack_app
    _configure_chat(harness)
    filename = _upload_sample(harness)
    _process(harness, filename)
    file_record = harness.repositories.files.get_file(filename)
    assert file_record["deletion_state"] == "active"

    if boundary == "disk":
        harness.storage.fail("delete")
    elif boundary == "postgres":
        harness.repositories.conversations.fail("delete_conversation_tree")
    elif boundary == "qdrant":
        harness.vector_store.fail("delete")
    elif boundary == "partial":
        harness.storage.fail("delete")

    response = harness.client.delete(f"/files/remove?path={filename}")

    assert response.status_code == 500
    assert response.json()["category"] == "document_delete_failed"
    assert response.json()["deletion_state"] == "delete_failed"
    assert response.json()["leftovers"]
    retained = harness.repositories.files.get_file(filename)
    assert retained["deletion_state"] == "delete_failed"
    assert retained["deletion_error"]
    assert retained["deletion_attempts"] >= 1
    if boundary == "partial":
        # Storage failed but the vector cleanup still ran: a partial
        # failure makes progress where it can without claiming success.
        assert (
            harness.qdrant.count(
                collection_name=harness.collection_name,
                exact=True,
            ).count
            == 0
        )
        assert harness.storage.list() != []

    if boundary == "disk":
        harness.storage.unfail("delete")
    elif boundary == "postgres":
        harness.repositories.conversations.unfail("delete_conversation_tree")
    elif boundary == "qdrant":
        harness.vector_store.unfail("delete")
    elif boundary == "partial":
        harness.storage.unfail("delete")

    retry = harness.client.delete(f"/files/remove?path={filename}")
    assert retry.status_code == 200
    assert harness.repositories.files.get_file(filename) is None
    assert harness.storage.list() == []
    assert (
        harness.qdrant.count(
            collection_name=harness.collection_name,
            exact=True,
        ).count
        == 0
    )
    with harness.session_factory() as session:
        assert session.query(Conversation).count() == 0
        assert session.query(Turn).count() == 0
        assert session.query(Source).count() == 0


@pytest.mark.full_stack
def test_chat_and_process_are_blocked_while_deletion_failed(full_stack_app):
    """A failed deletion blocks new chat and ingestion until retried."""
    harness = full_stack_app
    _configure_chat(harness)
    filename = _upload_sample(harness)
    _process(harness, filename)

    harness.vector_store.fail("delete")
    assert harness.client.delete(f"/files/remove?path={filename}").status_code == 500
    harness.vector_store.unfail("delete")

    chat = harness.client.post(
        "/response", json={"filename": filename, "query": "What changed?"}
    )
    assert chat.status_code == 409
    assert chat.json()["category"] == "document_delete_failed"

    process = harness.client.post("/process-file", json={"filename": filename})
    assert process.status_code == 409
    assert process.json()["category"] == "document_delete_failed"

    assert harness.client.delete(f"/files/remove?path={filename}").status_code == 200


@pytest.mark.full_stack
def test_http_cancellation_covers_queued_and_interrupted_running_jobs(full_stack_app):
    """Cancellation is durable across the HTTP, Postgres, and worker restart path."""
    harness = full_stack_app
    filename = _upload_sample(harness)

    queued_cancel = harness.client.post(f"/ingestion-jobs/{filename}/cancel")
    assert queued_cancel.status_code == 200
    assert queued_cancel.json()["job"]["state"] == "cancelled"

    retry = harness.client.post(f"/ingestion-jobs/{filename}/retry")
    assert retry.status_code == 201
    claimed = harness.repositories.ingestion_jobs.claim_next("worker-cancelled")
    assert claimed is not None
    running_cancel = harness.client.post(f"/ingestion-jobs/{filename}/cancel")
    assert running_cancel.status_code == 200
    assert running_cancel.json()["job"]["state"] == "cancelling"

    with harness.session_factory() as session, session.begin():
        record = session.get(IngestionJob, claimed["id"])
        record.heartbeat_at = record.heartbeat_at.replace(year=2000)

    assert harness.repositories.ingestion_jobs.recover_stale(0) == [claimed["id"]]
    assert (
        harness.client.get(f"/ingestion-jobs/{filename}").json()["job"]["state"]
        == "cancelling"
    )
    assert harness.drain("worker-cleanup") == 1
    final = harness.client.get(f"/ingestion-jobs/{filename}").json()["job"]
    assert final["state"] == "cancelled"
    assert final["worker_id"] is None


@pytest.mark.full_stack
def test_reindex_cancellation_keeps_the_ready_generation_queryable(full_stack_app):
    """A cancelled reindex leaves the active Qdrant generation untouched."""
    harness = full_stack_app
    _configure_chat(harness)
    filename = _upload_sample(harness)
    _process(harness, filename)
    before = harness.qdrant.count(
        collection_name=harness.collection_name, exact=True
    ).count
    assert before > 0

    assert harness.client.post(f"/ingestion-jobs/{filename}/retry").status_code == 201
    blocked = harness.client.post(
        "/response", json={"filename": filename, "query": "What changed?"}
    )
    assert blocked.status_code == 409
    assert blocked.json()["category"] == "document_indexing"

    cancelled = harness.client.post(f"/ingestion-jobs/{filename}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["job"]["state"] == "cancelled"
    after = harness.qdrant.count(
        collection_name=harness.collection_name, exact=True
    ).count
    assert after == before

    answer = harness.client.post(
        "/response", json={"filename": filename, "query": "What changed?"}
    )
    assert answer.status_code == 200


@pytest.mark.full_stack
def test_upload_reaches_ready_through_a_durable_job(full_stack_app):
    """Upload returns queued work and the worker finishes the index."""
    harness = full_stack_app
    filename = _upload_sample(harness)

    queued = harness.client.get(f"/ingestion-jobs/{filename}").json()["job"]
    assert queued["state"] == "queued"
    assert queued["progress"] == 0
    assert harness.embeddings.calls == [], "the request must not embed inline"
    assert (
        harness.client.post("/file/is-processed", json={"filename": filename}).json()[
            "is_processed"
        ]
        is False
    )

    assert (
        harness.client.post("/process-file", json={"filename": filename}).status_code
        == 202
    )
    assert harness.drain("worker-1") == 1

    ready = harness.client.get(f"/ingestion-jobs/{filename}").json()["job"]
    assert ready["state"] == "ready"
    assert ready["progress"] == 100
    assert ready["worker_id"] == "worker-1"
    assert ready["started_at"] and ready["finished_at"]
    assert ready["error_category"] is None
    assert (
        harness.client.post("/file/is-processed", json={"filename": filename}).json()[
            "is_processed"
        ]
        is True
    )

    points, _ = harness.qdrant.scroll(
        collection_name=harness.collection_name,
        limit=100,
        with_payload=True,
    )
    assert points, "the worker must index the parsed passages"
    assert {point.payload["pdf_name"] for point in points} == {filename}


@pytest.mark.full_stack
def test_worker_restart_recovers_an_interrupted_job(full_stack_app):
    """A job abandoned by a stopped worker is picked up by the next worker."""
    harness = full_stack_app
    filename = _upload_sample(harness)
    assert (
        harness.client.post("/process-file", json={"filename": filename}).status_code
        == 202
    )

    jobs = harness.repositories.ingestion_jobs
    claimed = jobs.claim_next("worker-stopped")
    assert claimed is not None
    assert claimed["state"] == "running"

    with jobs._session_factory() as session, session.begin():
        record = session.get(IngestionJob, claimed["id"])
        record.heartbeat_at = record.heartbeat_at.replace(year=2000)

    assert (
        harness.client.get(f"/ingestion-jobs/{filename}").json()["job"]["state"]
        == "running"
    ), "the running job stays visible while it is recoverable"

    assert harness.drain("worker-restarted") == 1

    recovered = harness.client.get(f"/ingestion-jobs/{filename}").json()["job"]
    assert recovered["state"] == "ready"
    assert recovered["worker_id"] == "worker-restarted"
    assert recovered["attempt"] == 1, "recovery continues the same attempt"
    assert jobs.count_active() == 0


@pytest.mark.full_stack
def test_postgres_allows_one_active_job_per_document(full_stack_app):
    """Racing workers claim each document once, and retries conflict."""
    harness = full_stack_app
    first_name = _upload_sample(harness)
    second_name = _upload_sample(harness)
    jobs = harness.repositories.ingestion_jobs
    first_file = harness.repositories.files.get_file(first_name)
    second_file = harness.repositories.files.get_file(second_name)

    # Four workers race for two queued documents. Each document must be
    # claimed exactly once: FOR UPDATE SKIP LOCKED plus the partial unique
    # index make a second claim of the same document impossible.
    claimed = _claim_in_parallel(jobs, "worker-1", "worker-2", "worker-3", "worker-4")
    winners = [job for job in claimed if job]
    assert len(winners) == 2
    assert {job["file_id"] for job in winners} == {
        first_file["id"],
        second_file["id"],
    }
    assert {job["state"] for job in winners} == {"running"}

    for job in winners:
        conflict = harness.client.post(f"/ingestion-jobs/{job['filename']}/retry")
        assert conflict.status_code == 409
        assert conflict.json()["category"] == "ingestion_job_active"
        assert jobs.get_active(job["file_id"])["id"] == job["id"]

    assert jobs.count_active() == 2

    for job in winners:
        assert jobs.mark_ready(job["id"], job["worker_id"]) is not None
    assert jobs.count_active() == 0

    for job in winners:
        assert (
            harness.client.post(f"/ingestion-jobs/{job['filename']}/retry").status_code
            == 201
        )
    assert harness.drain() == 2


@pytest.mark.full_stack
def test_real_document_goes_stale_and_reindexes_into_a_new_generation(
    full_stack_app, monkeypatch
):
    """A model change stops chat and a reindex replaces only the active index."""
    harness = full_stack_app
    _configure_chat(harness)
    filename = _upload_sample(harness)
    _process(harness, filename)

    def index_status() -> dict:
        response = harness.client.post(
            "/file/is-processed", json={"filename": filename}
        )
        assert response.status_code == 200, response.text
        return response.json()["index"]

    def generation_points(generation: int) -> int:
        return harness.qdrant.count(
            collection_name=harness.collection_name,
            count_filter={
                "must": [{"key": "index_generation", "match": {"value": generation}}]
            },
            exact=True,
        ).count

    assert index_status()["state"] == "ready"
    manifest = index_status()["manifest"]
    assert manifest["vector_dimension"] == 1024
    assert manifest["collection_name"] == harness.collection_name
    assert manifest["index_generation"] == 1
    assert manifest["parser"] == "false"
    assert manifest["reranker_model"] == harness.app_settings.rerank.rerank_model

    monkeypatch.setattr(harness.app_settings.embedding, "revision", "9f1c2ab")

    stale = index_status()
    assert stale["state"] == "stale"
    assert "embedding_revision" in stale["changes"]

    refused = harness.client.post(
        "/response", json={"query": "What is a RAG pipeline?", "filename": filename}
    )
    assert refused.status_code == 409
    body = refused.json()
    assert body["category"] == "index_stale"
    assert body["action"] == "reindex"
    assert [d["label"] for d in body["index"]["change_details"]] == [
        "embedding revision"
    ]
    # The vectors the user was reading are still there, untouched.
    assert generation_points(1) > 0

    reindex = harness.client.post(f"/files/{filename}/reindex")
    assert reindex.status_code == 201, reindex.text
    assert reindex.json()["job"]["generation"] == 2
    assert generation_points(1) > 0, "the active generation survives the reindex"

    assert harness.drain() == 1

    after = index_status()
    assert after["state"] == "ready"
    assert after["manifest"]["embedding_revision"] == "9f1c2ab"
    assert after["manifest"]["index_generation"] == 2
    assert generation_points(1) == 0, "the replaced generation is cleaned up"
    assert generation_points(2) > 0

    answered = harness.client.post(
        "/response", json={"query": "What is a RAG pipeline?", "filename": filename}
    )
    assert answered.status_code == 200, answered.text
    assert parse_sse(answered.text)[-1][1]["done"] is True


def _generation_points(harness: ApplicationHarness, filename: str, generation) -> int:
    """Count the Qdrant points of one generation for one document."""
    return harness.qdrant.count(
        collection_name=harness.collection_name,
        count_filter={
            "must": [
                {"key": "pdf_name", "match": {"value": filename}},
                {"key": "index_generation", "match": {"value": generation}},
            ]
        },
        exact=True,
    ).count


def _file_record(harness: ApplicationHarness, filename: str) -> dict:
    """Read the stored document record the activation transaction writes."""
    with harness.session_factory() as session:
        record = session.scalars(
            select(FileRecord).where(FileRecord.filename == filename)
        ).first()
        return {
            "index_generation": record.index_generation,
            "index_activated_at": record.index_activated_at,
            "index_manifest": record.index_manifest,
            "is_processed": record.is_processed,
        }


@pytest.mark.full_stack
def test_reindex_activates_a_validated_generation_and_records_the_transition(
    full_stack_app,
):
    """The replacement is validated, activated, and dated in one commit."""
    harness = full_stack_app
    _configure_chat(harness)
    filename = _upload_sample(harness)
    _process(harness, filename)
    first = _file_record(harness, filename)
    assert first["index_generation"] == 1
    assert first["index_activated_at"] is not None

    assert harness.client.post(f"/files/{filename}/reindex").status_code == 201
    assert _generation_points(harness, filename, 1) > 0
    assert harness.drain() == 1

    second = _file_record(harness, filename)
    assert second["index_generation"] == 2
    assert second["index_activated_at"] >= first["index_activated_at"]
    assert second["index_manifest"] != first["index_manifest"]
    assert _generation_points(harness, filename, 1) == 0
    assert _generation_points(harness, filename, 2) > 0
    assert harness.repositories.index_cleanups.list_pending() == []


@pytest.mark.full_stack
@pytest.mark.parametrize(
    "stage",
    ["open", "embed", "upsert", "count"],
    ids=["parsing", "embedding", "indexing", "validation"],
)
def test_a_reindex_failing_at_any_stage_keeps_the_previous_generation(
    full_stack_app, stage
):
    """A replacement that never completes leaves the readable index alone."""
    harness = full_stack_app
    _configure_chat(harness)
    filename = _upload_sample(harness)
    _process(harness, filename)
    before = _generation_points(harness, filename, 1)
    assert before > 0

    failures = {
        "open": harness.storage,
        "embed": harness.embeddings,
        "upsert": harness.vector_store,
        "count": harness.vector_store,
    }
    failures[stage].fail(stage)
    assert harness.client.post(f"/files/{filename}/reindex").status_code == 201
    assert harness.drain() == 1
    failures[stage].unfail(stage)

    job = harness.client.get(f"/ingestion-jobs/{filename}").json()["job"]
    assert job["state"] == "failed", stage
    record = _file_record(harness, filename)
    assert record["index_generation"] == 1, stage
    assert record["is_processed"] is True, stage
    assert _generation_points(harness, filename, 1) == before, stage
    answered = harness.client.post(
        "/response", json={"query": "What is a RAG pipeline?", "filename": filename}
    )
    assert answered.status_code == 200, answered.text


@pytest.mark.full_stack
def test_a_restarted_worker_finishes_the_recorded_generation_cleanup(full_stack_app):
    """A cleanup that failed once is retried from its durable record."""
    harness = full_stack_app
    _configure_chat(harness)
    filename = _upload_sample(harness)
    _process(harness, filename)

    harness.vector_store.fail("delete")
    assert harness.client.post(f"/files/{filename}/reindex").status_code == 201
    assert harness.drain() == 1

    assert _generation_points(harness, filename, 1) > 0
    assert _generation_points(harness, filename, 2) > 0
    pending = harness.repositories.index_cleanups.list_pending()
    assert [(task["filename"], task["generation"]) for task in pending] == [
        (filename, 1)
    ]
    # The replacement is already active, so chat never waits for the cleanup.
    assert _file_record(harness, filename)["index_generation"] == 2
    assert (
        harness.client.post(
            "/response", json={"query": "What is a RAG pipeline?", "filename": filename}
        ).status_code
        == 200
    )

    harness.vector_store.unfail("delete")
    assert harness.build_worker("worker-after-restart").run_pending_cleanups() == 1

    assert _generation_points(harness, filename, 1) == 0
    assert harness.repositories.index_cleanups.list_pending() == []


@pytest.mark.full_stack
def test_concurrent_reindex_requests_produce_one_activation(full_stack_app):
    """Racing reindex requests queue one generation, not two."""
    harness = full_stack_app
    _configure_chat(harness)
    filename = _upload_sample(harness)
    _process(harness, filename)
    responses: list = []
    lock = threading.Lock()
    barrier = threading.Barrier(3)

    def request_reindex() -> None:
        barrier.wait()
        response = harness.client.post(f"/files/{filename}/reindex")
        with lock:
            responses.append(response)

    threads = [threading.Thread(target=request_reindex) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert sorted(response.status_code for response in responses) == [201, 202]
    assert harness.drain() == 1
    assert _file_record(harness, filename)["index_generation"] == 2
    assert _generation_points(harness, filename, 1) == 0
    assert _generation_points(harness, filename, 2) > 0
    assert harness.repositories.index_cleanups.list_pending() == []


@pytest.mark.full_stack
def test_deleting_a_document_removes_every_generation_from_qdrant(full_stack_app):
    """Deletion leaves no points behind, whichever generation was active."""
    harness = full_stack_app
    _configure_chat(harness)
    filename = _upload_sample(harness)
    _process(harness, filename)
    assert harness.client.post(f"/files/{filename}/reindex").status_code == 201
    assert harness.drain() == 1
    assert _generation_points(harness, filename, 2) > 0

    assert harness.client.delete(f"/files/remove?path={filename}").status_code == 200

    assert (
        harness.qdrant.count(
            collection_name=harness.collection_name,
            count_filter={"must": [{"key": "pdf_name", "match": {"value": filename}}]},
            exact=True,
        ).count
        == 0
    )
    with harness.session_factory() as session:
        assert (
            session.scalars(
                select(FileRecord).where(FileRecord.filename == filename)
            ).first()
            is None
        )
    assert harness.repositories.index_cleanups.list_pending() == []


@pytest.mark.full_stack
def test_validation_rejects_a_generation_with_missing_vectors(full_stack_app):
    """Validation reads the stored points, not the counters of a write."""
    harness = full_stack_app
    _configure_chat(harness)
    filename = _upload_sample(harness)
    _process(harness, filename)
    service = VectorService(harness.vector_store)
    stored = _generation_points(harness, filename, 1)
    assert stored > 1

    assert service.validate_generation(filename, 1, stored)["total"] == stored

    harness.qdrant.delete(
        collection_name=harness.collection_name,
        points_selector=QdrantFilter(
            must=[
                FieldCondition(key="pdf_name", match=MatchValue(value=filename)),
                FieldCondition(key="index_generation", match=MatchValue(value=1)),
            ]
        ),
        wait=True,
    )

    with pytest.raises(VectorStoreConfigurationError) as raised:
        service.validate_generation(filename, 1, stored)
    assert "failed validation" in str(raised.value)


def _turns(harness: ApplicationHarness, filename: str) -> list[dict]:
    """Read one document's stored turns in sequence order."""
    file_id = harness.repositories.files.get_file(filename)["id"]
    conversation_id = harness.repositories.conversations.get_conversation_id(file_id)
    return harness.repositories.conversations.get_turns(conversation_id)


def _turn_status(harness: ApplicationHarness, filename: str) -> list[str]:
    """Read one document's stored turn outcomes."""
    return [turn["status"] for turn in _turns(harness, filename)]


@pytest.mark.full_stack
def test_postgres_allows_one_conversation_per_document(full_stack_app):
    """Racing indexing keeps one conversation, and a second insert is refused."""
    harness = full_stack_app
    filename = _upload_sample(harness)
    _process(harness, filename)
    file_id = harness.repositories.files.get_file(filename)["id"]
    repository = harness.repositories.conversations

    assert repository.ensure_conversation(file_id) == repository.get_conversation_id(
        file_id
    )
    with harness.session_factory() as session:
        assert session.query(Conversation).filter_by(file_id=file_id).count() == 1

    with pytest.raises(IntegrityError):
        with harness.session_factory() as session, session.begin():
            session.add(Conversation(file_id=file_id, id="f" * 32))

    # A reindex re-runs the same conversation creation without splitting it.
    assert harness.client.post(f"/files/{filename}/reindex").status_code == 201
    assert harness.drain() == 1
    with harness.session_factory() as session:
        assert session.query(Conversation).filter_by(file_id=file_id).count() == 1


@pytest.mark.full_stack
def test_concurrent_questions_become_distinct_ordered_turns(full_stack_app):
    """Four parallel questions get four sequences, not a tangled history."""
    harness = full_stack_app
    _configure_chat(harness)
    filename = _upload_sample(harness)
    _process(harness, filename)

    barrier = threading.Barrier(4)
    answers: list[str] = []
    lock = threading.Lock()

    def ask(index: int) -> None:
        barrier.wait()
        response = harness.client.post(
            "/response",
            json={"filename": filename, "query": f"What is stage {index}?"},
        )
        assert response.status_code == 200, response.text
        with lock:
            answers.append(
                "".join(
                    payload["text"]
                    for name, payload in parse_sse(response.text)
                    if name == "token"
                )
            )

    threads = [threading.Thread(target=ask, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert answers == ["The answer is grounded in the retrieved PDF text."] * 4
    turns = _turns(harness, filename)
    assert [turn["sequence"] for turn in turns] == [1, 2, 3, 4]
    assert {turn["status"] for turn in turns} == {"answered"}
    assert sorted(turn["question"] for turn in turns) == [
        f"What is stage {index}?" for index in range(4)
    ]

    history = harness.client.get(f"/messages?filename={filename}").json()["messages"]
    assert [message["text"] for message in history] == [
        value for turn in turns for value in (turn["question"], turn["answer"])
    ]


@pytest.mark.full_stack
def test_a_provider_failure_stores_a_failed_turn(full_stack_app):
    """A dead provider leaves a recorded failure, not a stranded question."""
    harness = full_stack_app
    _configure_chat(harness)
    filename = _upload_sample(harness)
    _process(harness, filename)
    harness.chat.fail("stream")

    response = harness.client.post(
        "/response", json={"filename": filename, "query": "What is a RAG pipeline?"}
    )

    assert [name for name, _ in parse_sse(response.text)] == ["error", "done"]
    turn = _turns(harness, filename)[0]
    assert turn["status"] == "failed"
    assert turn["question"] == "What is a RAG pipeline?"
    assert turn["answer"]
    assert turn["failure_reason"] == "provider failure"
    assert turn["completed_at"]
    history = harness.client.get(f"/messages?filename={filename}").json()["messages"]
    assert [message["turn_status"] for message in history] == ["failed", "failed"]


@pytest.mark.full_stack
def test_an_answer_that_postgres_cannot_save_closes_the_turn_as_failed(full_stack_app):
    """A database that refuses the answer still ends the turn as a failure."""
    harness = full_stack_app
    _configure_chat(harness)
    filename = _upload_sample(harness)
    _process(harness, filename)
    harness.repositories.conversations.fail("complete_turn")

    response = harness.client.post(
        "/response", json={"filename": filename, "query": "What is a RAG pipeline?"}
    )

    assert [name for name, _ in parse_sse(response.text)][-2:] == ["error", "done"]
    harness.repositories.conversations.unfail("complete_turn")
    turn = _turns(harness, filename)[0]
    assert turn["status"] == "failed"
    assert turn["question"] == "What is a RAG pipeline?"
    assert turn["failure_reason"] == "answer could not be saved"
    assert turn["completed_at"]
    history = harness.client.get(f"/messages?filename={filename}").json()["messages"]
    assert [message["turn_status"] for message in history] == ["failed", "failed"]


@pytest.mark.full_stack
def test_a_client_that_disconnects_mid_answer_cancels_its_turn(full_stack_app):
    """A closed stream ends the turn as cancelled instead of leaving it open."""
    harness = full_stack_app
    _configure_chat(harness)
    filename = _upload_sample(harness)
    _process(harness, filename)
    holding = harness.chat.hold_after_first_token()

    _reset_connection_after_first_token(
        harness.server.server_port,
        {"filename": filename, "query": "What is RAG?"},
    )
    harness.chat.release()

    assert holding.wait(timeout=30)
    assert _wait_for_turn_status(harness, filename, "cancelled")
    turn = _turns(harness, filename)[0]
    assert turn["question"] == "What is RAG?"
    assert turn["answer"] is None
    assert turn["failure_reason"] == "client disconnected"
    history = harness.client.get(f"/messages?filename={filename}").json()["messages"]
    assert [message["text"] for message in history] == ["What is RAG?"]


def _reset_connection_after_first_token(port: int, payload: dict) -> None:
    """Ask a question, read one token, then abort the connection."""
    # Closing a socket normally leaves the server free to finish writing into a
    # buffer nobody reads, so the reset is what the server actually notices.
    body = json.dumps(payload).encode()
    request = (
        "POST /response HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{port}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode() + body
    with socket.create_connection(("127.0.0.1", port), timeout=30) as connection:
        connection.sendall(request)
        received = b""
        while b"event: token" not in received:
            chunk = connection.recv(4096)
            if not chunk:
                raise AssertionError("the stream ended before the first token")
            received += chunk
        connection.setsockopt(
            socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
        )


def _wait_for_turn_status(
    harness: ApplicationHarness, filename: str, status: str, timeout: float = 15.0
) -> bool:
    """Wait until one document's first turn reaches the expected outcome."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _turn_status(harness, filename) == [status]:
            return True
        time.sleep(0.05)
    return False


@pytest.mark.full_stack
def test_an_unanswerable_history_state_reloads_as_it_was_accepted(full_stack_app):
    """A stored answer replays unchanged after the app is rebuilt."""
    harness = full_stack_app
    _configure_chat(harness)
    filename = _upload_sample(harness)
    _process(harness, filename)
    response = harness.client.post(
        "/response", json={"filename": filename, "query": "What is a RAG pipeline?"}
    )
    terminal = parse_sse(response.text)[-1][1]

    reloaded = harness.client.get(f"/messages?filename={filename}").json()["messages"]

    assert [message["sender"] for message in reloaded] == ["user", "bot"]
    assert reloaded[1]["text"] == "The answer is grounded in the retrieved PDF text."
    assert reloaded[1]["turn_status"] == "answered"
    assert reloaded[1]["turn_id"] == reloaded[0]["turn_id"]
    source_fields = ("content", "document", "chunk_index", "score", "page")
    assert [
        {field: source[field] for field in source_fields}
        for source in reloaded[1]["sources"]
    ] == [
        {field: source[field] for field in source_fields}
        for source in terminal["sources"]
    ]


@pytest.mark.full_stack
def test_the_turn_migration_keeps_every_legacy_exchange(legacy_database):
    """Legacy messages become ordered turns without losing a question or a source."""
    engine = legacy_database.engine
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO files (id, filename, title, is_processed) "
                "VALUES (:id, :filename, :title, true)"
            ),
            [
                {"id": "1" * 32, "filename": "one.pdf", "title": "One"},
                {"id": "2" * 32, "filename": "two.pdf", "title": "Two"},
                {"id": "3" * 32, "filename": "three.pdf", "title": "Three"},
            ],
        )
        # One document ended up with two conversations, each with one exchange.
        connection.execute(
            text(
                "INSERT INTO conversations (id, file_id, created_at) VALUES "
                "('a' || :one, :file, '2026-01-01'), "
                "('b' || :two, :file, '2026-01-02')"
            ),
            {"one": "1" * 31, "two": "2" * 31, "file": "1" * 32},
        )
        connection.execute(
            text(
                "INSERT INTO conversations (id, file_id) VALUES "
                "('c' || :one, :two), ('d' || :one, :three)"
            ),
            {"one": "3" * 31, "two": "2" * 32, "three": "3" * 32},
        )
        connection.execute(
            text(
                "INSERT INTO messages (id, conversation_id, sender, text, created_at) "
                "VALUES (:id, :conversation, :sender, :text, :created_at)"
            ),
            [
                {
                    "id": "m" + "1" * 31,
                    "conversation": "a" + "1" * 31,
                    "sender": "user",
                    "text": "What is a RAG pipeline?",
                    "created_at": "2026-01-01 10:00:00",
                },
                {
                    "id": "m" + "2" * 31,
                    "conversation": "a" + "1" * 31,
                    "sender": "bot",
                    "text": "It retrieves evidence for a generator.",
                    "created_at": "2026-01-01 10:00:05",
                },
                {
                    "id": "m" + "3" * 31,
                    "conversation": "b" + "2" * 31,
                    "sender": "user",
                    "text": "What does reranking do?",
                    "created_at": "2026-01-02 10:00:00",
                },
                {
                    "id": "m" + "4" * 31,
                    "conversation": "b" + "2" * 31,
                    "sender": "bot",
                    "text": "It reorders the candidates.",
                    "created_at": "2026-01-02 10:00:05",
                },
                {
                    "id": "m" + "5" * 31,
                    "conversation": "c" + "3" * 31,
                    "sender": "user",
                    "text": "Answered question",
                    "created_at": "2026-01-01 09:00:00",
                },
                {
                    "id": "m" + "6" * 31,
                    "conversation": "c" + "3" * 31,
                    "sender": "bot",
                    "text": "Answered reply",
                    "created_at": "2026-01-01 09:00:01",
                },
                {
                    "id": "m" + "7" * 31,
                    "conversation": "c" + "3" * 31,
                    "sender": "user",
                    "text": "Never answered",
                    "created_at": "2026-01-01 09:01:00",
                },
                {
                    "id": "m" + "8" * 31,
                    "conversation": "d" + "3" * 31,
                    "sender": "bot",
                    "text": "A reply no question asked for",
                    "created_at": "2026-01-01 08:00:00",
                },
            ],
        )
        connection.execute(
            text(
                "INSERT INTO sources "
                "(id, message_id, content, document, chunk_index, score) "
                "VALUES (:id, :message, 'Chunk', 'one.pdf', 0, 0.9)"
            ),
            [
                {"id": "s" + "1" * 31, "message": "m" + "2" * 31},
                {"id": "s" + "2" * 31, "message": "m" + "4" * 31},
            ],
        )

    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=BACKEND_DIR,
        env={**os.environ, "DATABASE_URL": legacy_database.url},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    with engine.connect() as connection:
        assert connection.scalar(text("SELECT version_num FROM alembic_version")) == (
            CURRENT_REVISION
        )
        assert (
            connection.scalar(
                text("SELECT count(*) FROM conversations WHERE file_id = :file"),
                {"file": "1" * 32},
            )
            == 1
        )
        rows = (
            connection.execute(
                text(
                    "SELECT t.sequence, t.question, t.answer, t.status, t.failure_reason, "
                    "f.filename, (SELECT count(*) FROM sources s "
                    "WHERE s.turn_id = t.id) AS source_count "
                    "FROM turns t "
                    "JOIN conversations c ON c.id = t.conversation_id "
                    "JOIN files f ON f.id = c.file_id "
                    "WHERE c.file_id = '11111111111111111111111111111111' "
                    "ORDER BY t.sequence"
                )
            )
            .mappings()
            .all()
        )
        assert [dict(row) for row in rows] == [
            {
                "sequence": 1,
                "question": "What is a RAG pipeline?",
                "answer": "It retrieves evidence for a generator.",
                "status": "answered",
                "failure_reason": None,
                "filename": "one.pdf",
                "source_count": 1,
            },
            {
                "sequence": 2,
                "question": "What does reranking do?",
                "answer": "It reorders the candidates.",
                "status": "answered",
                "failure_reason": None,
                "filename": "one.pdf",
                "source_count": 1,
            },
        ]
        stranded = (
            connection.execute(
                text(
                    "SELECT t.sequence, t.question, t.answer, t.status, t.failure_reason "
                    "FROM turns t JOIN conversations c ON c.id = t.conversation_id "
                    "WHERE c.file_id = '22222222222222222222222222222222' "
                    "ORDER BY t.sequence"
                )
            )
            .mappings()
            .all()
        )
        assert [dict(row) for row in stranded] == [
            {
                "sequence": 1,
                "question": "Answered question",
                "answer": "Answered reply",
                "status": "answered",
                "failure_reason": None,
            },
            {
                "sequence": 2,
                "question": "Never answered",
                "answer": None,
                "status": "unanswered",
                "failure_reason": "no answer was recorded",
            },
        ]
        orphan = (
            connection.execute(
                text(
                    "SELECT t.question, t.answer, t.status, t.failure_reason "
                    "FROM turns t JOIN conversations c ON c.id = t.conversation_id "
                    "WHERE c.file_id = '33333333333333333333333333333333'"
                )
            )
            .mappings()
            .all()
        )
        assert [dict(row) for row in orphan] == [
            {
                "question": None,
                "answer": "A reply no question asked for",
                "status": "answered",
                "failure_reason": "no question was recorded",
            }
        ]
