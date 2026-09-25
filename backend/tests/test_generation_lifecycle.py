"""
Index generation activation tests at the ingestion boundary.

A generation is built beside the active one, validated in full, activated in a
single transaction, and only then are the superseded vectors removed through a
durable cleanup that survives a restart and never touches the active
generation. Deleting a document removes every generation, and a worker that
finishes late cannot reactivate one.
"""

import io
from dataclasses import replace
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import create_app
from composition import Services
from db import Base, FileRecord
from repositories import build_repositories
from services.accounts.secrets_service import encrypt_api_key
from services.ingestion.worker import IngestionWorker
from services.parsing.document_parser import Chunk
from services.retrieval.base import RetrievalResult
from services.retrieval.hybrid import build_sparse_vector
from services.retrieval.vector_service import FETCH_K, shape_sources
from tests.ingestion_helpers import build_test_worker
from tests.sse import parse_sse

PDF_BYTES = b"%PDF-fake-bytes"


@pytest.fixture
def session_factory():
    """Create the schema in one in-memory database and return its sessions."""
    engine = create_engine(
        "sqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


@pytest.fixture
def repositories(session_factory):
    """Build all aggregate repositories over the in-memory database."""
    repos = build_repositories(session_factory)
    repos.app_settings.upsert_app_settings(
        "groq", "openai/gpt-oss-120b", encrypt_api_key("sk-test-chat-key")
    )
    return repos


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
            {"name": name, "size": len(blob)}
            for name, blob in sorted(self.blobs.items())
        ]

    def url(self, filename):
        """Do url."""
        return f"/storage/{filename}"


class FakeVectorService:
    """Vector service that records every generation written and removed."""

    def __init__(self):
        """Initialize."""
        self.upserts = []
        self.deleted_generations = []
        self.deleted_unversioned = []
        self.deleted = []
        self.fail_generation_deletions = 0
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
        self.upserts.append((embeddings, chunks, filename, kwargs.get("generation")))

    def query_vectors(
        self, embedding, filename, top_k=FETCH_K, query_text=None, **kwargs
    ):
        """Return the recorded matches shaped like production retrieval."""
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

    def delete_by_generation(self, filename, generation):
        """Delete one generation, optionally failing to simulate an outage."""
        if self.fail_generation_deletions > 0:
            self.fail_generation_deletions -= 1
            raise RuntimeError("vector store is unavailable")
        self.deleted_generations.append((filename, generation))

    def delete_unversioned(self, filename):
        """Delete vectors stored before generations existed."""
        self.deleted_unversioned.append(filename)

    def delete_by_filename(self, filename):
        """Delete every generation of one document."""
        self.deleted.append(filename)

    def delete_all(self):
        """Do delete all."""


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
        ]


class FailingParser(FakeParser):
    """Parser that fails, so a replacement index never becomes active."""

    def get_chunk_objects(self, filename, file_bytes):
        """Fail before returning chunks."""
        raise RuntimeError("parser is unavailable")


class DeletingVectorService(FakeVectorService):
    """Vector service that loses the document mid-write, like a concurrent user."""

    def __init__(self):
        """Initialize."""
        super().__init__()
        self.on_upsert = None
        self._deleted = False

    def upsert_chunks(self, embeddings, chunks, filename, **kwargs):
        """Write the generation, then delete the document it was written for."""
        result = super().upsert_chunks(embeddings, chunks, filename, **kwargs)
        if self.on_upsert is not None and not self._deleted:
            self._deleted = True
            self.on_upsert()
        return result


class FakeEmbeddingService:
    """Fixed-vector stand-in for the embedding service."""

    def __init__(self):
        """Initialize."""
        self.embedded = []

    def embed_texts(self, texts):
        """Do embed texts."""
        if isinstance(texts, str):
            texts = [texts]
        self.embedded.extend(texts)
        return [[0.1, 0.2] for _ in texts]


class FakeChatFactory:
    """Chat provider factory that streams a fixed answer."""

    def __call__(self, credentials):
        """Build a provider streaming a fixed answer."""

        class Provider:
            def stream_response(self, query, context):
                yield "The answer is 42."

        return Provider()


def build_test_app(repositories, settings_obj, vectors, storage=None):
    """Compose the app like production over the given vector service."""
    storage = storage or FakeStorage()
    services = replace(
        Services.from_settings(settings_obj),
        repositories=repositories,
        storage=storage,
        parser=FakeParser(),
        embedding_service=FakeEmbeddingService(),
        vector_service=vectors,
        chat_provider_factory=FakeChatFactory(),
    )
    application = create_app(settings_obj, services=services)
    application.config.update(
        TEST_REPOSITORIES=repositories,
        TEST_STORAGE=storage,
        TEST_PARSER=FakeParser(),
        TEST_EMBEDDINGS=FakeEmbeddingService(),
        TEST_VECTORS=vectors,
    )
    return application


@pytest.fixture
def app(repositories, settings_obj):
    """Compose the app like production, with fakes wired through the factory."""
    return build_test_app(repositories, settings_obj, FakeVectorService())


@pytest.fixture
def client(app):
    """Do client."""
    with app.test_client() as client:
        yield client


def upload(client, name="doc.pdf"):
    """Upload a document and return its stored filename."""
    data = {"file": (io.BytesIO(PDF_BYTES), name)}
    return client.post(
        "/upload", data=data, content_type="multipart/form-data"
    ).get_json()["file"]["name"]


def index_document(client, app, name="doc.pdf"):
    """Upload a document and run its ingestion job to ready."""
    filename = upload(client, name)
    build_test_worker(app).drain()
    return filename


def reindex(client, app, filename):
    """Queue a reindex and run it to completion."""
    assert client.post(f"/files/{filename}/reindex").status_code == 201
    build_test_worker(app).drain()


def ask(client, filename, query="What is the answer?"):
    """Ask a question and return the response."""
    return client.post("/response", json={"query": query, "filename": filename})


def test_activation_records_the_transition_time_and_manifest(client, app):
    """Becoming active stores when it happened next to the manifest."""
    filename = index_document(client, app)

    record = app.config["TEST_REPOSITORIES"].files.get_file(filename)

    assert record["index_activated_at"] is not None
    assert record["index_manifest"]
    assert record["index_generation"] == 1

    first_activation = record["index_activated_at"]
    reindex(client, app, filename)

    replaced = app.config["TEST_REPOSITORIES"].files.get_file(filename)
    assert replaced["index_generation"] == 2
    assert replaced["index_activated_at"] >= first_activation


def test_successful_reindex_removes_the_superseded_generation(client, app):
    """The replaced generation is cleaned up once the new one is active."""
    filename = index_document(client, app)
    vectors = app.config["TEST_VECTORS"]

    reindex(client, app, filename)

    assert vectors.deleted_generations == [(filename, 1)]
    assert app.config["TEST_REPOSITORIES"].index_cleanups.list_pending() == []
    assert ask(client, filename).status_code == 200


def test_cleanup_is_durable_and_retried_after_a_restart(client, app):
    """A cleanup that fails stays recorded and completes on the next run."""
    filename = index_document(client, app)
    vectors = app.config["TEST_VECTORS"]
    vectors.fail_generation_deletions = 1
    repositories = app.config["TEST_REPOSITORIES"]

    reindex(client, app, filename)

    assert vectors.deleted_generations == []
    pending = repositories.index_cleanups.list_pending()
    assert [(task["filename"], task["generation"]) for task in pending] == [
        (filename, 1)
    ]
    assert pending[0]["attempts"] == 1
    # The new generation is already active, so chat never waits for cleanup.
    assert repositories.files.get_file(filename)["index_generation"] == 2
    assert ask(client, filename).status_code == 200

    build_test_worker(app).drain()

    assert vectors.deleted_generations == [(filename, 1)]
    assert repositories.index_cleanups.list_pending() == []


def test_a_repeated_removal_is_idempotent(client, app):
    """Running the same recorded removal twice leaves one clean outcome."""
    filename = index_document(client, app)
    reindex(client, app, filename)
    vectors = app.config["TEST_VECTORS"]
    repositories = app.config["TEST_REPOSITORIES"]
    vectors.deleted_generations.clear()
    repositories.index_cleanups.schedule(
        repositories.files.get_file(filename)["id"], filename, 1
    )

    worker = build_test_worker(app)
    assert worker.run_pending_cleanups() == 1
    assert worker.run_pending_cleanups() == 0

    # Only the superseded generation is touched, and only while it is recorded.
    assert vectors.deleted_generations == [(filename, 1)]
    assert repositories.index_cleanups.list_pending() == []
    assert repositories.files.get_file(filename)["index_generation"] == 2
    assert ask(client, filename).status_code == 200


def test_a_removal_recorded_before_activation_stays_pending(client, app):
    """A restart between recording a removal and activating it loses nothing."""
    filename = index_document(client, app)
    vectors = app.config["TEST_VECTORS"]
    repositories = app.config["TEST_REPOSITORIES"]
    # The window a crash lands in: the removal is written down while the
    # generation it replaces is still the active one.
    repositories.index_cleanups.schedule(
        repositories.files.get_file(filename)["id"], filename, 1
    )

    assert build_test_worker(app).run_pending_cleanups() == 0

    assert vectors.deleted_generations == []
    assert [
        task["generation"] for task in repositories.index_cleanups.list_pending()
    ] == [1]

    reindex(client, app, filename)

    assert vectors.deleted_generations == [(filename, 1)]
    assert repositories.index_cleanups.list_pending() == []


def test_cleanup_waits_while_a_job_still_holds_the_generation(client, app):
    """A generation a queued or running job still owns is not removed."""
    filename = index_document(client, app)
    vectors = app.config["TEST_VECTORS"]
    repositories = app.config["TEST_REPOSITORIES"]
    assert client.post(f"/files/{filename}/reindex").status_code == 201
    generation = repositories.ingestion_jobs.get_latest(filename)["generation"]
    repositories.index_cleanups.schedule(
        repositories.files.get_file(filename)["id"], filename, generation
    )

    build_test_worker(app).run_pending_cleanups()

    assert vectors.deleted_generations == []
    assert [
        task["generation"] for task in repositories.index_cleanups.list_pending()
    ] == [generation]


def test_deleting_a_document_removes_every_generation(client, app):
    """Deletion clears the active and the superseded generations together."""
    filename = index_document(client, app)
    vectors = app.config["TEST_VECTORS"]
    reindex(client, app, filename)

    assert client.delete(f"/files/remove?path={filename}").status_code == 200

    assert vectors.deleted == [filename]
    assert app.config["TEST_REPOSITORIES"].files.get_file(filename) is None
    assert app.config["TEST_REPOSITORIES"].index_cleanups.list_pending() == []


def test_a_worker_that_finishes_after_deletion_cannot_activate(
    client, app, repositories, settings_obj
):
    """A generation written after the delete is removed and never activated."""
    filename = upload(client)
    vectors = DeletingVectorService()
    racing_app = build_test_app(
        repositories,
        settings_obj,
        vectors,
        storage=app.config["TEST_STORAGE"],
    )

    with racing_app.test_client() as racing_client:
        vectors.on_upsert = lambda: racing_client.delete(
            f"/files/remove?path={filename}"
        )
        build_test_worker(racing_app).drain()

    assert repositories.files.get_file(filename) is None
    assert vectors.upserts[-1][3] == 1
    # The deletion cleared the document, and the late write was cleaned up
    # after it rather than left behind.
    assert vectors.deleted == [filename]
    assert vectors.deleted_generations == [(filename, 1)]
    assert repositories.ingestion_jobs.get_latest(filename) is None


def test_activation_refuses_a_document_that_started_deleting(
    app, session_factory, repositories
):
    """The activation transaction checks the document is still there."""
    record, _job = repositories.files.create_file_with_job("doc.pdf")
    filename = record["filename"]
    repositories.files.mark_deleting(filename)

    job = repositories.ingestion_jobs.claim_next("worker-1")
    assert job is not None

    assert (
        repositories.ingestion_jobs.mark_ready(
            job["id"], "worker-1", index_generation=1, index_manifest="{}"
        )
        is None
    )

    with session_factory() as session:
        record = session.scalars(
            select(FileRecord).where(FileRecord.filename == filename)
        ).first()
        assert record.is_processed is False
        assert record.index_generation is None
    assert repositories.ingestion_jobs.get(job["id"])["state"] != "ready"


def test_a_cancelled_reindex_leaves_the_previous_generation_chat_ready(
    client, app, repositories
):
    """Cancelling a replacement keeps the index the user was reading."""
    filename = index_document(client, app)
    vectors = app.config["TEST_VECTORS"]
    assert client.post(f"/files/{filename}/reindex").status_code == 201
    assert client.post(f"/ingestion-jobs/{filename}/cancel").status_code == 200

    assert build_test_worker(app).drain() == 0

    assert repositories.ingestion_jobs.get_latest(filename)["state"] == "cancelled"
    assert repositories.files.get_file(filename)["index_generation"] == 1
    assert vectors.upserts[-1][3] == 1
    assert ask(client, filename).status_code == 200


def test_a_limited_reindex_leaves_the_previous_generation_chat_ready(
    client, app, repositories
):
    """A replacement stopped by a resource limit never becomes active."""
    filename = index_document(client, app)
    vectors = app.config["TEST_VECTORS"]
    assert client.post(f"/files/{filename}/reindex").status_code == 201
    limited_worker = IngestionWorker(
        repositories=repositories,
        storage=app.config["TEST_STORAGE"],
        parser=app.config["TEST_PARSER"],
        embedding_service=app.config["TEST_EMBEDDINGS"],
        vector_service=vectors,
        worker_id="worker-limit",
        limits=SimpleNamespace(max_pages=0),
    )

    assert limited_worker.drain() == 1

    job = repositories.ingestion_jobs.get_latest(filename)
    assert job["state"] == "failed"
    assert job["error_category"] == "page_limit_exceeded"
    assert repositories.files.get_file(filename)["index_generation"] == 1
    assert vectors.upserts[-1][3] == 1
    assert ask(client, filename).status_code == 200


def test_a_failed_reindex_leaves_the_document_stale_when_the_runtime_moved_on(
    client, app, settings_obj, monkeypatch
):
    """The previous generation is kept, and the document stays stale."""
    filename = index_document(client, app)
    monkeypatch.setattr(settings_obj.chunking, "chunk_size_tokens", 1024)
    assert client.post(f"/files/{filename}/reindex").status_code == 201
    app.config["TEST_PARSER"] = FailingParser()
    build_test_worker(app).drain()

    monkeypatch.setattr(settings_obj.embedding, "revision", "9f1c2ab")

    index = client.post("/file/is-processed", json={"filename": filename}).get_json()[
        "index"
    ]
    assert index["state"] == "stale"
    assert "embedding_revision" in index["changes"]
    # The generation is still there, and a reindex is what brings it back.
    assert (
        app.config["TEST_REPOSITORIES"].files.get_file(filename)["index_generation"]
        == 1
    )
    response = ask(client, filename)
    assert response.status_code == 409
    body = response.get_json()
    assert body["category"] == "index_stale"
    assert body["action"] == "reindex"


def test_chat_never_reads_a_generation_being_rebuilt(client, app):
    """A running reindex keeps the active generation readable."""
    filename = index_document(client, app)
    assert client.post(f"/files/{filename}/reindex").status_code == 201

    response = ask(client, filename)

    if response.status_code == 200:
        assert parse_sse(response.get_data(as_text=True))[-1][1]["done"] is True
    else:
        body = response.get_json()
        assert body["category"] == "document_indexing"
        assert body["job"]["state"] == "queued"
