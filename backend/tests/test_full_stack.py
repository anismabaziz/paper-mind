"""Full-stack tests over HTTP with real infrastructure and deterministic models."""

import os
import pathlib
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass

import psycopg
import pytest
from psycopg import sql
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from db import Conversation, IngestionJob, Message, Source
from services.embeddings.local_embeddings import EmbeddingService
from services.llm.base import LLMProvider
from services.retrieval.base import VectorStore
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
        "ingestion_jobs",
        "messages",
        "sources",
    }
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT version_num FROM alembic_version")) == (
            "f2b4c6d8a013"
        )
        for table in (
            "app_settings",
            "conversations",
            "files",
            "ingestion_jobs",
            "messages",
            "sources",
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
        assert harness.qdrant.count(
            collection_name=harness.collection_name,
            exact=True,
        ).count == 0
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
        assert session.query(Message).count() == 0
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
def test_upload_reaches_ready_through_a_durable_job(full_stack_app):
    """Upload returns queued work and the worker finishes the index."""
    harness = full_stack_app
    filename = _upload_sample(harness)

    queued = harness.client.get(f"/ingestion-jobs/{filename}").json()["job"]
    assert queued["state"] == "queued"
    assert queued["progress"] == 0
    assert harness.embeddings.calls == [], "the request must not embed inline"
    assert (
        harness.client.post(
            "/file/is-processed", json={"filename": filename}
        ).json()["is_processed"]
        is False
    )

    assert harness.client.post(
        "/process-file", json={"filename": filename}
    ).status_code == 202
    assert harness.drain("worker-1") == 1

    ready = harness.client.get(f"/ingestion-jobs/{filename}").json()["job"]
    assert ready["state"] == "ready"
    assert ready["progress"] == 100
    assert ready["worker_id"] == "worker-1"
    assert ready["started_at"] and ready["finished_at"]
    assert ready["error_category"] is None
    assert (
        harness.client.post(
            "/file/is-processed", json={"filename": filename}
        ).json()["is_processed"]
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
    assert harness.client.post(
        "/process-file", json={"filename": filename}
    ).status_code == 202

    jobs = harness.repositories.ingestion_jobs
    claimed = jobs.claim_next("worker-stopped")
    assert claimed is not None
    assert claimed["state"] == "running"

    with jobs._session_factory() as session, session.begin():
        record = session.get(IngestionJob, claimed["id"])
        record.heartbeat_at = record.heartbeat_at.replace(year=2000)

    assert harness.client.get(f"/ingestion-jobs/{filename}").json()["job"][
        "state"
    ] == "running", "the running job stays visible while it is recoverable"

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
    claimed = _claim_in_parallel(
        jobs, "worker-1", "worker-2", "worker-3", "worker-4"
    )
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
        assert harness.client.post(f"/ingestion-jobs/{job['filename']}/retry").status_code == 201
    assert harness.drain() == 2
