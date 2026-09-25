"""Database-backed worker that processes queued ingestion jobs."""

import inspect
import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from repositories.ingestion_jobs import IngestionJobRepository
from services.embeddings.local_embeddings import EmbeddingService
from services.indexing.manifest import manifest_builder as manifest_builder_for
from services.ingestion.limits import (
    IngestionCancelled,
    IngestionLimitExceeded,
    ResourceGuard,
    ResourceLimits,
    current_memory_bytes,
    estimate_output_bytes,
)
from services.parsing.document_parser import DocumentIngestor
from services.retrieval.base import (
    VectorDimensionError,
    VectorStoreConfigurationError,
    VectorStoreUnavailableError,
)
from services.retrieval.vector_service import VectorService
from settings import get_settings
from storage import LocalStorage

log = logging.getLogger(__name__)

DEFAULT_STALE_TIMEOUT_S = 300
HEARTBEAT_INTERVAL_S = 15.0


def _page_count(parser: DocumentIngestor, filename: str, file_content: bytes) -> int:
    page_count = getattr(parser, "get_page_count", None)
    if callable(page_count):
        return page_count(filename, file_content)
    return 0


def _page_count_from_chunks(chunks: list[Any]) -> int:
    pages = [chunk.page_no for chunk in chunks if getattr(chunk, "page_no", None)]
    return max((int(page) for page in pages), default=0)


def _call_with_check(
    method: Callable[..., Any],
    *args: Any,
    check: Callable[[], None],
    **kwargs: Any,
) -> Any:
    """Call an adapter with cooperative checks when its interface accepts them."""
    try:
        parameters = inspect.signature(method).parameters.values()
        accepts_check = any(
            parameter.name == "check" or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
    except (TypeError, ValueError):
        accepts_check = False
    if accepts_check:
        return method(*args, **kwargs, check=check)
    return method(*args, **kwargs)


@dataclass(frozen=True)
class JobError(Exception):
    """A safe ingestion failure category with a user-facing message."""

    category: str
    message: str


def _classify(error: Exception) -> JobError:
    """Map a failure to one safe category and actionable message."""
    if isinstance(error, IngestionLimitExceeded):
        return JobError(error.category, error.message)
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
        usage_provider: Callable[[], dict[str, int | float]] | None = None,
    ) -> None:
        """Bind the heartbeat to one running job."""
        self._jobs = jobs
        self._job_id = job_id
        self._worker_id = worker_id
        self._interval_s = interval_s
        self._usage_provider = usage_provider
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
                usage = (
                    self._usage_provider() if self._usage_provider is not None else None
                )
                if (
                    self._jobs.heartbeat(
                        self._job_id,
                        self._worker_id,
                        usage=usage,
                    )
                    is None
                ):
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
        limits: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
        memory_reader: Callable[[], int] = current_memory_bytes,
        manifest_builder: Callable[[bytes, int], Any] | None = None,
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
        self._limits = ResourceLimits.from_value(limits)
        self._clock = clock
        self._memory_reader = memory_reader
        # The manifest records the configuration this worker indexes with, so
        # the process that serves chat can tell when it no longer matches.
        self._manifest_builder = manifest_builder or manifest_builder_for(
            get_settings()
        )

    @property
    def worker_id(self) -> str:
        """Return the identifier that owns running jobs."""
        return self._worker_id

    def drain(self, limit: int | None = None) -> int:
        """Process queued jobs until the queue is empty or the limit is hit."""
        processed = 0
        while limit is None or processed < limit:
            job = self._jobs.claim_next(
                self._worker_id,
                self._stale_timeout_s,
                limits=self._limits.as_dict(),
            )
            if job is None:
                break
            self._process(job)
            processed += 1
            if (self._jobs.get(job["id"]) or {}).get("state") == "cancelling":
                break
        return processed

    def _stage(
        self,
        job: dict[str, Any],
        stage: str,
        progress: int,
        guard: ResourceGuard | None = None,
        cleanup: bool = False,
    ) -> bool:
        """Persist stage and progress; finish a requested cancellation."""
        usage = guard.snapshot() if guard is not None else None
        updated = self._jobs.heartbeat(
            job["id"], self._worker_id, stage, progress, usage=usage
        )
        if updated is not None:
            return True
        current = self._jobs.get(job["id"])
        if current is None:
            if cleanup:
                self._cleanup_partial(job)
            return False
        self._cancel_if_requested(job, guard, cleanup=cleanup)
        if cleanup and current.get("state") in ("stale", "cancelled"):
            self._cleanup_partial(job)
        return False

    def _cancel_if_requested(
        self,
        job: dict[str, Any],
        guard: ResourceGuard | None = None,
        cleanup: bool = False,
    ) -> bool:
        """Finish a cancellation request if the worker still owns the job."""
        current = self._jobs.get(job["id"])
        if not current or current.get("state") != "cancelling":
            return False
        if cleanup and not self._cleanup_partial(job):
            self._jobs.release_cancellation(job["id"], self._worker_id)
            return True
        self._jobs.mark_cancelled(
            job["id"],
            self._worker_id,
            usage=guard.snapshot() if guard is not None else None,
        )
        return True

    def _cooperative_check(self, job: dict[str, Any], guard: ResourceGuard) -> None:
        """Check limits and cancellation between units of expensive work."""
        guard.observe()
        current = self._jobs.get(job["id"])
        if (
            current is None
            or current.get("state") != "running"
            or current.get("worker_id") != self._worker_id
        ):
            raise IngestionCancelled

    def _cleanup_partial(self, job: dict[str, Any]) -> bool:
        """Remove vectors written for a job that will not become active."""
        current = self._jobs.get(job["id"])
        if current and current.get("worker_id") not in (None, self._worker_id):
            return False
        if current and current.get("state") in ("queued", "ready", "failed"):
            return True
        generation = job.get("generation")
        delete_generation = getattr(self._vectors, "delete_by_generation", None)
        try:
            if callable(delete_generation) and generation is not None:
                delete_generation(job["filename"], generation)
                return True
            file_record = self._files.get_file(job["filename"])
            if not file_record or not file_record.get("is_processed"):
                self._vectors.delete_by_filename(job["filename"])
            return True
        except Exception as cleanup_error:
            log.warning(
                "partial ingestion cleanup failed for %s: %s",
                job["filename"],
                cleanup_error,
            )
            return False

    def _delete_previous_generation(
        self, filename: str, generation: int | None
    ) -> bool:
        """Remove an older generation after the replacement is active."""
        if generation is None:
            delete_legacy = getattr(self._vectors, "delete_unversioned", None)
        else:
            delete_legacy = getattr(self._vectors, "delete_by_generation", None)
        if not callable(delete_legacy):
            return False
        try:
            if generation is None:
                delete_legacy(filename)
            else:
                delete_legacy(filename, generation)
            return True
        except Exception as cleanup_error:
            log.warning(
                "previous generation cleanup failed for %s: %s",
                filename,
                cleanup_error,
            )
            return False

    def _process(self, job: dict[str, Any]) -> None:
        """Run one claimed job through the ingestion stages."""
        filename = job["filename"]
        usage = job.get("usage") or {}
        try:
            elapsed_offset = float(usage.get("elapsed_seconds", 0.0))
        except (TypeError, ValueError):
            elapsed_offset = 0.0
        guard = ResourceGuard(
            self._limits,
            self._clock,
            self._memory_reader,
            elapsed_offset=elapsed_offset,
        )
        partial_vectors = False
        with _Heartbeat(
            self._jobs,
            job["id"],
            self._worker_id,
            usage_provider=guard.snapshot,
        ):
            try:
                if job.get("state") == "cancelling" or job.get("cancel_requested_at"):
                    self._cancel_if_requested(job, guard, cleanup=True)
                    return
                file_record = self._files.get_file(filename)
                if file_record is None:
                    raise FileNotFoundError(filename)
                if file_record.get("deletion_state") in ("deleting", "delete_failed"):
                    self._supersede(job)
                    return
                previous_generation = file_record.get("index_generation")
                had_ready_index = bool(file_record.get("is_processed"))
                guard.observe()
                if not self._stage(job, "parsing", 10, guard):
                    return
                file_content = self._storage.open(filename)
                page_count = _page_count(self._parser, filename, file_content)
                if page_count < 0:
                    raise ValueError("Page count unavailable")
                guard.observe(page_count=page_count)
                chunks = _call_with_check(
                    self._parser.get_chunk_objects,
                    filename,
                    file_content,
                    check=lambda: self._cooperative_check(job, guard),
                )
                if not chunks:
                    raise ValueError("No text extracted from document")
                page_count = max(page_count, _page_count_from_chunks(chunks))
                if page_count == 0 and file_content.startswith(b"%PDF"):
                    raise ValueError("Page count unavailable")
                text_bytes = sum(len(chunk.text.encode("utf-8")) for chunk in chunks)
                guard.observe(
                    page_count=page_count,
                    extracted_text_bytes=text_bytes,
                )
                if not self._stage(job, "embedding", 45, guard):
                    return
                texts = [chunk.text for chunk in chunks]
                embeddings = _call_with_check(
                    self._embeddings.embed_texts,
                    texts,
                    check=lambda: self._cooperative_check(job, guard),
                )
                if len(embeddings) != len(chunks):
                    raise ValueError("Embedding count does not match parsed passages")
                output_bytes = estimate_output_bytes(embeddings, chunks)
                guard.observe(output_bytes=output_bytes)
                if not self._stage(job, "indexing", 75, guard):
                    return
                if not had_ready_index and not callable(
                    getattr(self._vectors, "delete_by_generation", None)
                ):
                    self._vectors.delete_by_filename(filename)
                partial_vectors = True
                _call_with_check(
                    self._vectors.upsert_chunks,
                    embeddings,
                    chunks,
                    filename,
                    generation=job["generation"],
                    check=lambda: self._cooperative_check(job, guard),
                )
                if not self._stage(job, "validating", 90, guard, cleanup=True):
                    return
                if isinstance(self._vectors, VectorService):
                    self._vectors.validate_generation(
                        filename,
                        job["generation"],
                        len(embeddings),
                    )
                guard.observe()
                if self._cancel_if_requested(job, guard, cleanup=True):
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
                if (
                    self._jobs.mark_ready(
                        job["id"],
                        self._worker_id,
                        index_generation=job["generation"],
                        index_manifest=self._manifest_builder(
                            file_content, job["generation"]
                        ).to_json(),
                    )
                    is None
                ):
                    self._cancel_if_requested(job, guard, cleanup=True)
                    log.warning(
                        "ingestion job %s lost its claim before ready",
                        job["id"],
                    )
                    return
                self._delete_previous_generation(filename, previous_generation)
                log.info("ingestion job %s ready for %s", job["id"], filename)
            except IngestionCancelled:
                self._cancel_if_requested(job, guard, cleanup=partial_vectors)
                if partial_vectors and self._jobs.get(job["id"]) is None:
                    self._cleanup_partial(job)
            except JobError as exc:
                self._fail(job, exc, guard)
            except IngestionLimitExceeded as exc:
                self._fail(job, _classify(exc), guard)
            except Exception as exc:  # noqa: BLE001 - persisted as a safe category
                log.exception("ingestion job %s failed for %s", job["id"], filename)
                self._fail(job, _classify(exc), guard)

    def _supersede(self, job: dict[str, Any]) -> None:
        """Stop a job whose document is being deleted."""
        self._jobs.mark_superseded(
            job["id"],
            self._worker_id,
            "document_deleting",
            "The document is being deleted, so indexing was stopped.",
        )
        self._cleanup_partial(job)

    def _fail(
        self,
        job: dict[str, Any],
        error: JobError,
        guard: ResourceGuard | None = None,
    ) -> None:
        """Persist a failed job and undo partially written vectors."""
        filename = job["filename"]
        current = self._jobs.get(job["id"])
        if current and current.get("state") == "cancelling":
            self._cancel_if_requested(job, guard, cleanup=True)
            return
        file_record = self._files.get_file(filename)
        had_ready_index = bool(file_record and file_record.get("is_processed"))
        self._cleanup_partial(job)
        if not had_ready_index:
            try:
                self._files.set_processed(filename, False)
            except Exception as state_error:
                log.warning(
                    "processed reset after failed ingestion for %s failed: %s",
                    filename,
                    state_error,
                )
        self._jobs.mark_failed(
            job["id"],
            self._worker_id,
            error.category,
            error.message,
            usage=guard.snapshot() if guard is not None else None,
        )
