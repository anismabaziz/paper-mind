"""
Durable ingestion-job tests.

Uploading a document creates one durable ingestion job and returns without
waiting for parsing or embedding. A database-backed worker claims jobs so
only one active job processes a document at a time, persists stage and
progress, recovers interrupted work, and retries without duplicating
vectors.
"""

import io
from dataclasses import replace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import create_app
from composition import Services
from db import Base, IngestionJob
from repositories import build_repositories
from services.accounts.secrets_service import encrypt_api_key
from services.ingestion.worker import IngestionWorker
from services.parsing.document_parser import Chunk


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


class FakeVectorService:
    """Vector service stand-in that records writes without indexing."""

    def __init__(self):
        """Initialize."""
        self.upserts = []
        self.deleted = []

    def upsert_chunks(self, embeddings, chunks, filename, **kwargs):
        """Do upsert chunks."""
        self.upserts.append((embeddings, chunks, filename))

    def delete_by_filename(self, filename):
        """Do delete by filename."""
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


@pytest.fixture
def app(repositories, settings_obj):
    """Compose the app like production, with fakes wired through the factory."""
    storage = FakeStorage()
    parser = FakeParser()
    vectors = FakeVectorService()
    embeddings = FakeEmbeddingService()
    services = replace(
        Services.from_settings(settings_obj),
        repositories=repositories,
        storage=storage,
        parser=parser,
        embedding_service=embeddings,
        vector_service=vectors,
    )
    application = create_app(settings_obj, services=services)
    application.config["ingestion_storage"] = storage
    application.config["ingestion_parser"] = parser
    application.config["ingestion_vectors"] = vectors
    application.config["ingestion_embeddings"] = embeddings
    return application


def build_worker(app, repositories, parser, worker_id="test-worker"):
    """Build a worker over the app's injected dependencies."""
    return IngestionWorker(
        repositories=repositories,
        storage=app.config["ingestion_storage"],
        parser=parser,
        embedding_service=app.config["ingestion_embeddings"],
        vector_service=app.config["ingestion_vectors"],
        worker_id=worker_id,
    )


@pytest.fixture
def worker(app, repositories):
    """Build a worker over the same injected dependencies as the app."""
    return build_worker(app, repositories, app.config["ingestion_parser"])


@pytest.fixture
def client(app):
    """Do client."""
    with app.test_client() as client:
        yield client


def upload(client, name="doc.pdf"):
    """Do upload."""
    data = {"file": (io.BytesIO(b"%PDF-fake-bytes"), name)}
    return client.post("/upload", data=data, content_type="multipart/form-data")


def test_upload_creates_a_queued_job_without_processing(client, app):
    """Upload returns a queued job before parsing or embedding runs."""
    response = upload(client)

    assert response.status_code == 200
    body = response.get_json()
    filename = body["file"]["name"]
    job = body["job"]
    assert job["state"] == "queued"
    assert job["filename"] == filename
    assert job["attempt"] == 1
    assert job["progress"] == 0

    assert app.config["ingestion_embeddings"].embedded == []
    assert app.config["ingestion_vectors"].upserts == []

    status = client.get(f"/ingestion-jobs/{filename}").get_json()["job"]
    assert status["state"] == "queued"
    assert status["stage"] == "queued"

    listing = client.get("/files").get_json()["files"]
    assert listing[0]["ingestion"]["state"] == "queued"
    assert (
        client.post("/file/is-processed", json={"filename": filename}).get_json()[
            "is_processed"
        ]
        is False
    )


def test_worker_claims_a_job_and_records_stages_until_ready(client, worker, app):
    """A worker claims queued work and persists each stage to ready."""
    filename = upload(client).get_json()["file"]["name"]

    assert worker.drain() == 1
    assert worker.drain() == 0

    job = client.get(f"/ingestion-jobs/{filename}").get_json()["job"]
    assert job["state"] == "ready"
    assert job["stage"] == "ready"
    assert job["progress"] == 100
    assert job["worker_id"] == "test-worker"
    assert job["started_at"] and job["finished_at"]
    assert job["heartbeat_at"]
    assert job["error_category"] is None

    vectors = app.config["ingestion_vectors"]
    assert [upsert[2] for upsert in vectors.upserts] == [filename]
    assert app.config["ingestion_embeddings"].embedded
    assert (
        client.post("/file/is-processed", json={"filename": filename}).get_json()[
            "is_processed"
        ]
        is True
    )


class FailingParser(FakeParser):
    """Parser stand-in that fails on demand."""

    def __init__(self, error):
        """Bind the failure to raise."""
        self.error = error
        self.calls = 0

    def get_chunk_objects(self, filename, file_bytes):
        """Fail before returning chunks."""
        self.calls += 1
        raise self.error


def test_failed_job_persists_a_safe_error_category_and_allows_retry(
    client, app, repositories
):
    """A failed job keeps a safe category, an attempt count, and a retry path."""
    filename = upload(client).get_json()["file"]["name"]
    failing = build_worker(
        app,
        repositories,
        FailingParser(RuntimeError("secret token abc123 leaked here")),
        worker_id="worker-a",
    )

    assert failing.drain() == 1

    job = repositories.ingestion_jobs.get_latest(filename)
    assert job["state"] == "failed"
    assert job["error_category"] == "ingestion_failed"
    assert "secret token" not in (job["error_message"] or "")
    assert job["progress"] < 100
    assert job["finished_at"]
    assert (
        client.post("/file/is-processed", json={"filename": filename}).get_json()[
            "is_processed"
        ]
        is False
    )

    retry = client.post(f"/ingestion-jobs/{filename}/retry")
    assert retry.status_code == 201
    retried = retry.get_json()["job"]
    assert retried["state"] == "queued"
    assert retried["attempt"] == 2
    assert retried["error_category"] is None
    assert retried["generation"] > job["generation"]


def test_only_one_active_job_processes_a_document(client, app, repositories):
    """A second claim and a second retry cannot duplicate active work."""
    filename = upload(client).get_json()["file"]["name"]
    jobs = repositories.ingestion_jobs
    file_record = repositories.files.get_file(filename)

    first_claim = jobs.claim_next("worker-a")
    assert first_claim is not None
    assert first_claim["filename"] == filename

    other_claim = jobs.claim_next("worker-b")
    assert other_claim is None, "a running job must not be claimed twice"

    retry = client.post(f"/ingestion-jobs/{filename}/retry")
    assert retry.status_code == 409
    assert retry.get_json()["category"] == "ingestion_job_active"
    assert retry.get_json()["job"]["worker_id"] == "worker-a"

    jobs.mark_ready(first_claim["id"], "worker-a")
    queued = client.post(f"/ingestion-jobs/{filename}/retry")
    assert queued.status_code == 201
    assert queued.get_json()["job"]["state"] == "queued"
    assert file_record["is_processed"] is False


def test_a_superseded_queued_job_is_marked_stale(client, app, repositories):
    """A newer attempt marks the older queued job stale instead of running it."""
    filename = upload(client).get_json()["file"]["name"]
    jobs = repositories.ingestion_jobs
    first = jobs.get_latest(filename)
    assert first["state"] == "queued"

    retry = client.post(f"/ingestion-jobs/{filename}/retry")
    assert retry.status_code == 201
    second = retry.get_json()["job"]
    assert second["state"] == "queued"
    assert second["attempt"] == 2
    assert second["generation"] == 2

    rows = {row["id"]: row for row in _all_job_rows(jobs)}
    assert rows[first["id"]]["state"] == "stale"
    assert rows[first["id"]]["stage"] == "stale"
    assert rows[second["id"]]["state"] == "queued"
    assert jobs.count_active() == 1


def _all_job_rows(jobs):
    """Return every persisted job row keyed by id."""
    with jobs._session_factory() as session:
        return [
            {"id": record.id, "state": record.state, "stage": record.stage}
            for record in session.query(IngestionJob).all()
        ]


def test_retry_after_a_failed_job_reprocesses_without_duplicate_vectors(
    client, app, repositories
):
    """A retried job rebuilds one index and does not leave two active jobs."""
    filename = upload(client).get_json()["file"]["name"]
    vectors = app.config["ingestion_vectors"]
    failing = build_worker(
        app,
        repositories,
        FailingParser(RuntimeError("parse failed")),
        worker_id="worker-a",
    )
    assert failing.drain() == 1
    assert vectors.deleted == [filename]

    assert client.post(f"/ingestion-jobs/{filename}/retry").status_code == 201
    healthy = build_worker(app, repositories, FakeParser(), worker_id="worker-b")

    assert healthy.drain() == 1
    assert healthy.drain() == 0

    job = repositories.ingestion_jobs.get_latest(filename)
    assert job["state"] == "ready"
    assert job["attempt"] == 2
    assert repositories.ingestion_jobs.count_active() == 0
    assert [upsert[2] for upsert in vectors.upserts] == [filename]


def test_interrupted_worker_leaves_a_recoverable_job(client, app, repositories):
    """A job whose worker stopped is requeued and finishes after a restart."""
    filename = upload(client).get_json()["file"]["name"]
    jobs = repositories.ingestion_jobs

    interrupted = jobs.claim_next("worker-a")
    assert interrupted is not None
    assert interrupted["state"] == "running"

    with jobs._session_factory() as session, session.begin():
        record = session.get(IngestionJob, interrupted["id"])
        record.heartbeat_at = record.heartbeat_at.replace(year=2000)

    restarted = build_worker(app, repositories, FakeParser(), worker_id="worker-b")

    assert restarted.drain() == 1

    job = jobs.get_latest(filename)
    assert job["state"] == "ready"
    assert job["worker_id"] == "worker-b"
    assert job["attempt"] == 1, "recovery continues the same attempt"
    assert job["progress"] == 100
    assert jobs.count_active() == 0
    assert [upsert[2] for upsert in app.config["ingestion_vectors"].upserts] == [
        filename
    ]

