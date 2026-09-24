"""Database-backed worker that processes queued ingestion jobs."""

import logging
import threading
import uuid
from dataclasses import dataclass
from typing import Any

from repositories.ingestion_jobs import IngestionJobRepository
from services.embeddings.local_embeddings import EmbeddingService
from services.parsing.document_parser import DocumentIngestor
from services.retrieval.base import (
    VectorDimensionError,
    VectorStoreConfigurationError,
    VectorStoreUnavailableError,
)
from services.retrieval.vector_service import VectorService
from storage import LocalStorage

log = logging.getLogger(__name__)

DEFAULT_STALE_TIMEOUT_S = 300
HEARTBEAT_INTERVAL_S = 15.0


@dataclass(frozen=True)
class JobError(Exception):
    """A safe ingestion failure category with a user-facing message."""

    category: str
    message: str


def _classify(error: Exception) -> JobError:
    """Map a failure to one safe category and actionable message."""
    if isinstance(error, VectorStoreUnavailableError):
        return JobError(
            "vector_store_unavailable",
            "The vector store is unavailable. Retry indexing when it is running.",
        )
    if isinstance(error, VectorStoreConfigurationError):
        return JobError(
            "vector_store_configuration",
            "The vector store configuration is invalid. Check the server settings.",
        )
    if isinstance(error, VectorDimensionError):
        return JobError(
            "vector_dimension_mismatch",
            "Stored vectors use a different dimension. Retry after clearing vectors.",
        )
    if isinstance(error, FileNotFoundError):
        return JobError(
            "file_missing",
            "The document file is missing. Upload it again before retrying.",
        )
    if isinstance(error, PermissionError):
        return JobError("permission_denied", "The document could not be read.")
    if isinstance(error, ValueError):
        return JobError("invalid_document", "The document could not be parsed.")
    return JobError(
        "ingestion_failed",
        "Indexing failed. Retry, or check the server logs for details.",
    )


class _Heartbeat:
    """Refresh a running job's heartbeat while a long stage is in flight."""

    def __init__(
        self,
        jobs: IngestionJobRepository,
        job_id: str,
        worker_id: str,
        interval_s: float = HEARTBEAT_INTERVAL_S,
    ) -> None:
        """Bind the heartbeat to one running job."""
        self._jobs = jobs
        self._job_id = job_id
        self._worker_id = worker_id
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_Heartbeat":
        """Start refreshing the heartbeat in the background."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        """Stop refreshing the heartbeat."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        """Beat until the stage finishes or the claim is lost."""
        while not self._stop.wait(self._interval_s):
            try:
                if self._jobs.heartbeat(self._job_id, self._worker_id) is None:
                    return
            except Exception:  # noqa: BLE001 - a beat failure must not kill work
                log.warning("ingestion heartbeat failed for job %s", self._job_id)


class IngestionWorker:
    """Claim queued jobs and run parsing, embedding, and indexing durably."""

    def __init__(
        self,
        repositories: Any,
        storage: LocalStorage,
        parser: DocumentIngestor,
        embedding_service: EmbeddingService,
        vector_service: VectorService,
        worker_id: str | None = None,
        stale_timeout_s: int = DEFAULT_STALE_TIMEOUT_S,
    ) -> None:
        """Bind a worker to the dependencies it needs to process a job."""
        self._repositories = repositories
        self._jobs = repositories.ingestion_jobs
        self._files = repositories.files
        self._conversations = repositories.conversations
        self._storage = storage
        self._parser = parser
        self._embeddings = embedding_service
        self._vectors = vector_service
        self._worker_id = worker_id or uuid.uuid4().hex
        self._stale_timeout_s = stale_timeout_s

    @property
    def worker_id(self) -> str:
        """Return the identifier that owns running jobs."""
        return self._worker_id

    def drain(self, limit: int | None = None) -> int:
        """Process queued jobs until the queue is empty or the limit is hit."""
        processed = 0
        while limit is None or processed < limit:
            job = self._jobs.claim_next(self._worker_id, self._stale_timeout_s)
            if job is None:
                break
            self._process(job)
            processed += 1
        return processed

    def _stage(self, job: dict[str, Any], stage: str, progress: int) -> bool:
        """Persist stage and progress; return whether the job is still owned."""
        return (
            self._jobs.heartbeat(job["id"], self._worker_id, stage, progress)
            is not None
        )

    def _process(self, job: dict[str, Any]) -> None:
        """Run one claimed job through the ingestion stages."""
        filename = job["filename"]
        with _Heartbeat(self._jobs, job["id"], self._worker_id):
            try:
                file_record = self._files.get_file(filename)
                if file_record is None:
                    raise FileNotFoundError(filename)
                if file_record.get("deletion_state") in ("deleting", "delete_failed"):
                    self._supersede(job)
                    return
                if not self._stage(job, "parsing", 10):
                    return
                file_content = self._storage.open(filename)
                chunks = self._parser.get_chunk_objects(filename, file_content)
                if not chunks:
                    raise ValueError("No text extracted from document")
                if not self._stage(job, "embedding", 45):
                    return
                embeddings = self._embeddings.embed_texts(
                    [chunk.text for chunk in chunks]
                )
                if not self._stage(job, "indexing", 75):
                    return
                # Clear the document's previous vectors first so a retry
                # rewrites one index instead of appending a second copy of
                # every passage.
                self._vectors.delete_by_filename(filename)
                self._vectors.upsert_chunks(embeddings, chunks, filename)
                if not self._stage(job, "validating", 90):
                    return
                fresh = self._files.get_file(filename)
                if fresh is None or fresh.get("deletion_state") in (
                    "deleting",
                    "delete_failed",
                ):
                    self._supersede(job)
                    return
                if not fresh.get("is_processed") and not (
                    self._conversations.get_conversation_id(fresh["id"])
                ):
                    self._conversations.create_conversation(fresh["id"])
                self._files.set_processed(filename, True)
                if self._jobs.mark_ready(job["id"], self._worker_id) is None:
                    log.warning(
                        "ingestion job %s lost its claim before ready",
                        job["id"],
                    )
                    return
                log.info("ingestion job %s ready for %s", job["id"], filename)
            except JobError as exc:
                self._fail(job, exc)
            except Exception as exc:  # noqa: BLE001 - persisted as a safe category
                log.exception("ingestion job %s failed for %s", job["id"], filename)
                self._fail(job, _classify(exc))

    def _supersede(self, job: dict[str, Any]) -> None:
        """Stop a job whose document is being deleted."""
        self._jobs.mark_superseded(
            job["id"],
            self._worker_id,
            "document_deleting",
            "The document is being deleted, so indexing was stopped.",
        )

    def _fail(self, job: dict[str, Any], error: JobError) -> None:
        """Persist a failed job and undo partially written vectors."""
        filename = job["filename"]
        try:
            self._vectors.delete_by_filename(filename)
        except Exception as cleanup_error:  # noqa: BLE001 - best effort cleanup
            log.warning(
                "vector cleanup after failed ingestion for %s failed: %s",
                filename,
                cleanup_error,
            )
        try:
            self._files.set_processed(filename, False)
        except Exception as state_error:  # noqa: BLE001 - best effort state reset
            log.warning(
                "processed reset after failed ingestion for %s failed: %s",
                filename,
                state_error,
            )
        self._jobs.mark_failed(
            job["id"], self._worker_id, error.category, error.message
        )
