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
from types import SimpleNamespace

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
        self.generations = []
        self.deleted_generations = []

    def upsert_chunks(self, embeddings, chunks, filename, **kwargs):
        """Do upsert chunks."""
        self.upserts.append((embeddings, chunks, filename))
        self.generations.append(kwargs.get("generation"))

    def delete_by_generation(self, filename, generation):
        """Delete one generation."""
        self.deleted_generations.append((filename, generation))

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


class UnknownPageParser(FakeParser):
    """Parser that cannot establish a safe page count."""

    def get_page_count(self, filename, file_bytes):
        """Report an unavailable page count."""
        return -1


class CancellingParser(FakeParser):
    """Parser that requests cancellation before returning chunks."""

    def __init__(self, cancel):
        """Bind the cancellation callback."""
        self.cancel = cancel

    def get_chunk_objects(self, filename, file_bytes):
        """Request cancellation and return normal chunks."""
        self.cancel()
        return super().get_chunk_objects(filename, file_bytes)


class CancellingEmbedding(FakeEmbeddingService):
    """Embedding adapter that requests cancellation during its call."""

    def __init__(self, cancel):
        """Bind the cancellation callback."""
        super().__init__()
        self.cancel = cancel

    def embed_texts(self, texts):
        """Request cancellation and return normal embeddings."""
        self.cancel()
        return super().embed_texts(texts)


class CancellingVector(FakeVectorService):
    """Vector adapter that requests cancellation after writing."""

    def __init__(self, cancel):
        """Bind the cancellation callback."""
        super().__init__()
        self.cancel = cancel

    def upsert_chunks(self, embeddings, chunks, filename, **kwargs):
        """Write normally and then request cancellation."""
        result = super().upsert_chunks(embeddings, chunks, filename, **kwargs)
        self.cancel()
        return result


class CleanupFailingVector(CancellingVector):
    """Vector adapter that can defer partial-generation cleanup."""

    def __init__(self, cancel):
        """Bind cancellation and start cleanup in a failed state."""
        super().__init__(cancel)
        self.fail_cleanup = True

    def delete_by_generation(self, filename, generation):
        """Fail cleanup until the test clears the switch."""
        if self.fail_cleanup:
            raise RuntimeError("cleanup unavailable")
        return super().delete_by_generation(filename, generation)


class TwoChunkParser(FakeParser):
    """Parser that returns two passages."""

    def get_chunk_objects(self, filename, file_bytes):
        """Return two deterministic passages."""
        return [
            super().get_chunk_objects(filename, file_bytes)[0],
            Chunk(
                text="second chunk",
                page_no=1,
                chunk_index=1,
                content_hash="hash-1",
            ),
        ]


class ShortEmbedding(FakeEmbeddingService):
    """Embedding adapter that returns too few vectors."""

    def embed_texts(self, texts):
        """Return one vector regardless of the input count."""
        return [[0.1, 0.2]]


class DeletingVector(FakeVectorService):
    """Vector adapter that deletes its document during an upsert."""

    def __init__(self, delete):
        """Bind the deletion callback."""
        super().__init__()
        self.delete = delete

    def upsert_chunks(self, embeddings, chunks, filename, **kwargs):
        """Write vectors and then delete the document metadata."""
        result = super().upsert_chunks(embeddings, chunks, filename, **kwargs)
        self.delete()
        return result


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


def test_a_live_cancellation_cannot_be_claimed_by_another_worker(client, repositories):
    """A second worker leaves a running cancellation with its current owner."""
    filename = upload(client).get_json()["file"]["name"]
    jobs = repositories.ingestion_jobs
    claimed = jobs.claim_next("worker-owner")
    assert claimed is not None
    assert client.post(f"/ingestion-jobs/{filename}/cancel").status_code == 200
    assert jobs.claim_next("worker-other") is None
    assert jobs.get_latest(filename)["worker_id"] == "worker-owner"


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
    assert vectors.deleted_generations == [(filename, 1)]

    assert client.post(f"/ingestion-jobs/{filename}/retry").status_code == 201
    healthy = build_worker(app, repositories, FakeParser(), worker_id="worker-b")

    assert healthy.drain() == 1
    assert healthy.drain() == 0

    job = repositories.ingestion_jobs.get_latest(filename)
    assert job["state"] == "ready"
    assert job["attempt"] == 2
    assert repositories.ingestion_jobs.count_active() == 0
    assert [upsert[2] for upsert in vectors.upserts] == [filename]


def test_recovered_job_keeps_elapsed_time_for_the_limit(client, app, repositories):
    """A restarted attempt cannot reset the job's elapsed-time budget."""
    filename = upload(client).get_json()["file"]["name"]

    jobs = repositories.ingestion_jobs
    claimed = jobs.claim_next("worker-old")
    assert claimed is not None
    with jobs._session_factory() as session, session.begin():
        record = session.get(IngestionJob, claimed["id"])
        record.usage_json = '{"elapsed_seconds": 5}'
        record.heartbeat_at = record.heartbeat_at.replace(year=2000)
    assert jobs.recover_stale(stale_timeout_s=0) == [claimed["id"]]

    worker = IngestionWorker(
        repositories=repositories,
        storage=app.config["ingestion_storage"],
        parser=app.config["ingestion_parser"],
        embedding_service=app.config["ingestion_embeddings"],
        vector_service=app.config["ingestion_vectors"],
        worker_id="worker-new",
        limits=SimpleNamespace(max_elapsed_seconds=4),
        clock=lambda: 0.0,
        memory_reader=lambda: 0,
    )
    assert worker.drain() == 1
    job = jobs.get_latest(filename)
    assert job["state"] == "failed"
    assert job["error_category"] == "time_limit_exceeded"


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


def test_queued_job_cancellation_is_idempotent(client, repositories):
    """A queued job can be cancelled without changing its terminal row twice."""
    filename = upload(client).get_json()["file"]["name"]
    jobs = repositories.ingestion_jobs

    first = client.post(f"/ingestion-jobs/{filename}/cancel")
    second = client.post(f"/ingestion-jobs/{filename}/cancel")

    assert first.status_code == 200
    assert second.status_code == 200
    first_job = first.get_json()["job"]
    second_job = second.get_json()["job"]
    assert first_job["id"] == second_job["id"]
    assert first_job["state"] == "cancelled"
    assert second_job["state"] == "cancelled"
    assert first_job["finished_at"]
    assert jobs.count_active() == 0


def test_cancellation_does_not_reactivate_terminal_jobs(client, app, repositories):
    """Terminal jobs stay terminal when cancellation is requested again."""
    filename = upload(client).get_json()["file"]["name"]
    healthy = build_worker(app, repositories, FakeParser(), worker_id="worker-ready")
    assert healthy.drain() == 1

    ready_cancel = client.post(f"/ingestion-jobs/{filename}/cancel")
    assert ready_cancel.status_code == 200
    assert ready_cancel.get_json()["job"]["state"] == "ready"
    assert repositories.ingestion_jobs.get_latest(filename)["state"] == "ready"

    retry = client.post(f"/ingestion-jobs/{filename}/retry")
    assert retry.status_code == 201
    cancelled = client.post(f"/ingestion-jobs/{filename}/cancel")
    assert cancelled.get_json()["job"]["state"] == "cancelled"
    assert client.post(f"/ingestion-jobs/{filename}/retry").status_code == 201
    assert (
        build_worker(
            app, repositories, FakeParser(), worker_id="worker-retry-ready"
        ).drain(limit=1)
        == 1
    )

    failed_name = upload(client, name="failed.pdf").get_json()["file"]["name"]
    failing = build_worker(
        app,
        repositories,
        FailingParser(RuntimeError("bad document")),
        worker_id="worker-failed",
    )
    assert failing.drain() == 1
    failed_cancel = client.post(f"/ingestion-jobs/{failed_name}/cancel")
    assert failed_cancel.get_json()["job"]["state"] == "failed"


def test_running_cancellation_stops_before_embedding(client, app, repositories):
    """A running job stops before embedding after cancellation."""
    filename = upload(client).get_json()["file"]["name"]
    jobs = repositories.ingestion_jobs
    cancel_response = {}

    def cancel_during_parse():
        cancel_response["value"] = client.post(f"/ingestion-jobs/{filename}/cancel")

    cancelling_worker = build_worker(
        app,
        repositories,
        CancellingParser(cancel_during_parse),
        worker_id="worker-cancel",
    )

    assert cancelling_worker.drain() == 1
    assert cancel_response["value"].status_code == 200
    assert cancel_response["value"].get_json()["job"]["state"] == "cancelling"

    job = jobs.get_latest(filename)
    assert job["state"] == "cancelled"
    assert job["stage"] == "cancelled"
    assert app.config["ingestion_embeddings"].embedded == []
    assert app.config["ingestion_vectors"].upserts == []
    assert jobs.count_active() == 0


def test_cancellation_during_embedding_does_not_start_indexing(
    client, app, repositories
):
    """Cancellation during embedding prevents vector writes."""
    filename = upload(client).get_json()["file"]["name"]
    embeddings = CancellingEmbedding(
        lambda: client.post(f"/ingestion-jobs/{filename}/cancel")
    )
    worker = IngestionWorker(
        repositories=repositories,
        storage=app.config["ingestion_storage"],
        parser=app.config["ingestion_parser"],
        embedding_service=embeddings,
        vector_service=app.config["ingestion_vectors"],
        worker_id="worker-cancel-embedding",
    )

    assert worker.drain() == 1
    job = repositories.ingestion_jobs.get_latest(filename)
    assert job["state"] == "cancelled"
    assert app.config["ingestion_vectors"].upserts == []
    assert repositories.files.get_file(filename)["is_processed"] is False


def test_cancellation_before_activation_finishes_the_job(
    client, app, repositories, monkeypatch
):
    """A cancellation racing the final database transition is not stranded."""
    filename = upload(client).get_json()["file"]["name"]

    jobs = repositories.ingestion_jobs

    def cancel_before_ready(job_id, worker_id, **kwargs):
        client.post(f"/ingestion-jobs/{filename}/cancel")
        return None

    monkeypatch.setattr(jobs, "mark_ready", cancel_before_ready)
    worker = build_worker(app, repositories, FakeParser(), worker_id="worker-race")
    assert worker.drain() == 1

    job = jobs.get_latest(filename)
    assert job["state"] == "cancelled"
    assert job["worker_id"] is None


def test_cancellation_during_upsert_cleans_the_partial_generation(
    client, app, repositories
):
    """Cancellation during upsert removes the unactivated generation."""
    filename = upload(client).get_json()["file"]["name"]
    vectors = CancellingVector(
        lambda: client.post(f"/ingestion-jobs/{filename}/cancel")
    )
    worker = IngestionWorker(
        repositories=repositories,
        storage=app.config["ingestion_storage"],
        parser=app.config["ingestion_parser"],
        embedding_service=app.config["ingestion_embeddings"],
        vector_service=vectors,
        worker_id="worker-cancel-upsert",
    )

    assert worker.drain() == 1
    job = repositories.ingestion_jobs.get_latest(filename)
    assert job["state"] == "cancelled"
    assert vectors.deleted_generations == [(filename, 1)]
    assert repositories.files.get_file(filename)["is_processed"] is False


def test_deletion_during_upsert_cleans_late_generation_vectors(
    client, app, repositories
):
    """A deletion racing an upsert cannot leave vectors behind."""
    filename = upload(client).get_json()["file"]["name"]

    vectors = DeletingVector(lambda: client.delete(f"/files/remove?path={filename}"))
    worker = IngestionWorker(
        repositories=repositories,
        storage=app.config["ingestion_storage"],
        parser=app.config["ingestion_parser"],
        embedding_service=app.config["ingestion_embeddings"],
        vector_service=vectors,
        worker_id="worker-delete-race",
    )

    assert worker.drain() == 1
    assert repositories.files.get_file(filename) is None
    assert (filename, 1) in vectors.deleted_generations


def test_failed_partial_cleanup_keeps_cancellation_recoverable(
    client, app, repositories
):
    """A cleanup outage leaves the job cancelling until cleanup can retry."""
    filename = upload(client).get_json()["file"]["name"]

    vectors = CleanupFailingVector(
        lambda: client.post(f"/ingestion-jobs/{filename}/cancel")
    )
    first_worker = IngestionWorker(
        repositories=repositories,
        storage=app.config["ingestion_storage"],
        parser=app.config["ingestion_parser"],
        embedding_service=app.config["ingestion_embeddings"],
        vector_service=vectors,
        worker_id="worker-cleanup-failure",
    )

    assert first_worker.drain() == 1
    assert repositories.ingestion_jobs.get_latest(filename)["state"] == "cancelling"

    vectors.fail_cleanup = False
    cleanup_worker = IngestionWorker(
        repositories=repositories,
        storage=app.config["ingestion_storage"],
        parser=app.config["ingestion_parser"],
        embedding_service=app.config["ingestion_embeddings"],
        vector_service=vectors,
        worker_id="worker-cleanup-retry",
    )
    assert cleanup_worker.drain() == 1
    assert repositories.ingestion_jobs.get_latest(filename)["state"] == "cancelled"


def test_embedding_count_mismatch_cannot_activate_an_incomplete_index(
    client, app, repositories
):
    """A short embedding result fails before the generation is written."""
    filename = upload(client).get_json()["file"]["name"]

    worker = IngestionWorker(
        repositories=repositories,
        storage=app.config["ingestion_storage"],
        parser=TwoChunkParser(),
        embedding_service=ShortEmbedding(),
        vector_service=app.config["ingestion_vectors"],
        worker_id="worker-short-embeddings",
    )

    assert worker.drain() == 1
    job = repositories.ingestion_jobs.get_latest(filename)
    assert job["state"] == "failed"
    assert job["error_category"] == "invalid_document"
    assert app.config["ingestion_vectors"].upserts == []


def test_unknown_page_count_fails_safe_before_parsing(client, app, repositories):
    """An unverifiable page count cannot bypass the page limit."""
    filename = upload(client).get_json()["file"]["name"]

    worker = build_worker(
        app, repositories, UnknownPageParser(), worker_id="worker-page-unknown"
    )

    assert worker.drain() == 1
    job = repositories.ingestion_jobs.get_latest(filename)
    assert job["state"] == "failed"
    assert job["error_category"] == "invalid_document"
    assert app.config["ingestion_embeddings"].embedded == []


def test_page_limit_fails_before_embedding_and_keeps_document_retryable(
    client, app, repositories
):
    """A page limit fails safely before the next expensive stage."""
    filename = upload(client).get_json()["file"]["name"]
    limits = SimpleNamespace(max_pages=0)
    limited_worker = IngestionWorker(
        repositories=repositories,
        storage=app.config["ingestion_storage"],
        parser=app.config["ingestion_parser"],
        embedding_service=app.config["ingestion_embeddings"],
        vector_service=app.config["ingestion_vectors"],
        worker_id="worker-limit",
        limits=limits,
    )

    assert limited_worker.drain() == 1
    job = repositories.ingestion_jobs.get_latest(filename)
    assert job["state"] == "failed"
    assert job["error_category"] == "page_limit_exceeded"
    assert app.config["ingestion_embeddings"].embedded == []
    assert app.config["ingestion_vectors"].upserts == []
    assert client.post(f"/ingestion-jobs/{filename}/retry").status_code == 201


class GrowingMemory:
    """Memory reader with a low baseline and a later increase."""

    def __init__(self):
        """Start before the measured increase."""
        self.calls = 0

    def __call__(self):
        """Return zero once, then a larger value."""
        self.calls += 1
        return 0 if self.calls == 1 else 2


class StepClock:
    """Monotonic test clock that advances on every read."""

    def __init__(self):
        """Start at zero."""
        self.ticks = 0

    def __call__(self):
        """Advance and return the clock value."""
        self.ticks += 1
        return float(self.ticks)


@pytest.mark.parametrize(
    ("limit_name", "limit", "expected_category", "clock", "memory_reader"),
    [
        ("max_extracted_text_bytes", 0, "text_limit_exceeded", None, None),
        ("max_output_bytes", 0, "output_limit_exceeded", None, None),
        ("max_elapsed_seconds", 0, "time_limit_exceeded", StepClock(), None),
        ("max_memory_bytes", 0, "memory_limit_exceeded", None, GrowingMemory()),
    ],
)
def test_each_resource_limit_fails_with_a_retryable_category(
    client,
    app,
    repositories,
    limit_name,
    limit,
    expected_category,
    clock,
    memory_reader,
):
    """Each resource limit records a safe retryable failure."""
    filename = upload(client).get_json()["file"]["name"]
    limits = SimpleNamespace(**{limit_name: limit})
    limited_worker = IngestionWorker(
        repositories=repositories,
        storage=app.config["ingestion_storage"],
        parser=app.config["ingestion_parser"],
        embedding_service=app.config["ingestion_embeddings"],
        vector_service=app.config["ingestion_vectors"],
        worker_id=f"worker-{limit_name}",
        limits=limits,
        clock=clock or (lambda: 0.0),
        memory_reader=memory_reader or (lambda: 0),
    )

    assert limited_worker.drain() == 1
    job = repositories.ingestion_jobs.get_latest(filename)
    assert job["state"] == "failed"
    assert job["error_category"] == expected_category
    assert job["usage"]
    assert client.post(f"/ingestion-jobs/{filename}/retry").status_code == 201


def test_limit_during_reindex_keeps_the_previous_ready_generation(
    client, app, repositories
):
    """A limited reindex leaves the active generation usable."""
    filename = upload(client).get_json()["file"]["name"]
    first_worker = build_worker(app, repositories, FakeParser(), worker_id="worker-one")
    assert first_worker.drain() == 1
    first = repositories.files.get_file(filename)
    assert first["is_processed"] is True
    assert first["index_generation"] == 1
    assert app.config["ingestion_vectors"].generations == [1]

    assert client.post(f"/ingestion-jobs/{filename}/retry").status_code == 201
    limited_worker = IngestionWorker(
        repositories=repositories,
        storage=app.config["ingestion_storage"],
        parser=app.config["ingestion_parser"],
        embedding_service=app.config["ingestion_embeddings"],
        vector_service=app.config["ingestion_vectors"],
        worker_id="worker-limit-reindex",
        limits=SimpleNamespace(max_pages=0),
    )
    assert limited_worker.drain() == 1

    failed = repositories.ingestion_jobs.get_latest(filename)
    current = repositories.files.get_file(filename)
    assert failed["state"] == "failed"
    assert failed["error_category"] == "page_limit_exceeded"
    assert current["is_processed"] is True
    assert current["index_generation"] == 1
    assert app.config["ingestion_vectors"].generations == [1]


def test_interrupted_cancelling_worker_leaves_a_cancelled_job(
    client, app, repositories
):
    """A stopped worker cannot strand a cancellation request."""
    filename = upload(client).get_json()["file"]["name"]
    jobs = repositories.ingestion_jobs
    claimed = jobs.claim_next("worker-interrupted")
    assert claimed is not None
    assert client.post(f"/ingestion-jobs/{filename}/cancel").status_code == 200

    with jobs._session_factory() as session, session.begin():
        record = session.get(IngestionJob, claimed["id"])
        record.heartbeat_at = record.heartbeat_at.replace(year=2000)

    assert jobs.recover_stale(stale_timeout_s=0) == [claimed["id"]]
    interrupted = jobs.get_latest(filename)
    assert interrupted["state"] == "cancelling"
    assert (
        build_worker(app, repositories, FakeParser(), worker_id="worker-cleanup").drain(
            limit=1
        )
        == 1
    )
    job = jobs.get_latest(filename)
    assert job["state"] == "cancelled"
    assert job["worker_id"] is None
