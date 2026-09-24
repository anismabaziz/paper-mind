"""Persistence for durable ingestion jobs."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from db import IngestionJob
from repositories.base import BaseRepository, to_record_dict

ACTIVE_STATES = ("queued", "running")


class JobConflictError(Exception):
    """One active ingestion job already processes the document."""


def _utcnow() -> datetime:
    """Return the current UTC time."""
    return datetime.now(timezone.utc)


def _age_s(timestamp: datetime | None) -> float | None:
    """Return how many seconds ago a timestamp was, across backends."""
    if timestamp is None:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return (_utcnow() - timestamp).total_seconds()


def ingestion_job_to_dict(record: IngestionJob) -> dict[str, Any]:
    """Convert a persisted job to a plain dictionary."""
    return to_record_dict(
        record,
        file_id=record.file_id,
        filename=record.filename,
        generation=record.generation,
        state=record.state,
        stage=record.stage,
        progress=record.progress,
        attempt=record.attempt,
        error_category=record.error_category,
        error_message=record.error_message,
        worker_id=record.worker_id,
        updated_at=record.updated_at.isoformat() if record.updated_at else None,
        started_at=record.started_at.isoformat() if record.started_at else None,
        finished_at=record.finished_at.isoformat() if record.finished_at else None,
        heartbeat_at=record.heartbeat_at.isoformat()
        if record.heartbeat_at
        else None,
    )


class IngestionJobRepository(BaseRepository):
    """Claim and track ingestion jobs for stored documents."""

    def enqueue(self, file_id: str, filename: str) -> dict[str, Any]:
        """Create one queued job; supersede older queued jobs as stale."""
        try:
            return self._enqueue(file_id, filename)
        except IntegrityError as error:
            # The partial unique index is the final arbiter when two callers
            # enqueue the same document at once.
            raise JobConflictError(
                "One active ingestion job already processes this document."
            ) from error

    def _enqueue(self, file_id: str, filename: str) -> dict[str, Any]:
        """Insert the queued job inside one transaction."""
        with self._session_factory() as session, session.begin():
            active = session.scalars(
                select(IngestionJob)
                .where(IngestionJob.file_id == file_id)
                .where(IngestionJob.state.in_(ACTIVE_STATES))
            ).all()
            for record in active:
                if record.state == "queued":
                    record.state = "stale"
                    record.stage = "stale"
                    record.finished_at = _utcnow()
            running = [record for record in active if record.state == "running"]
            if running:
                raise JobConflictError(
                    "One active ingestion job already processes this document."
                )
            latest = session.scalars(
                select(IngestionJob)
                .where(IngestionJob.file_id == file_id)
                .order_by(IngestionJob.generation.desc())
            ).first()
            generation = (latest.generation if latest else 0) + 1
            attempt = (latest.attempt if latest else 0) + 1
            record = IngestionJob(
                file_id=file_id,
                filename=filename,
                generation=generation,
                state="queued",
                stage="queued",
                progress=0,
                attempt=attempt,
            )
            session.add(record)
            session.flush()
            return ingestion_job_to_dict(record)

    def get_latest(self, filename: str) -> dict[str, Any] | None:
        """Return the newest job for a document, if any."""
        with self._session_factory() as session:
            record = session.scalars(
                select(IngestionJob)
                .where(IngestionJob.filename == filename)
                .order_by(IngestionJob.created_at.desc(), IngestionJob.generation.desc())
            ).first()
            return ingestion_job_to_dict(record) if record else None

    def get_active(self, file_id: str) -> dict[str, Any] | None:
        """Return the queued or running job for a document, if any."""
        with self._session_factory() as session:
            record = session.scalars(
                select(IngestionJob)
                .where(IngestionJob.file_id == file_id)
                .where(IngestionJob.state.in_(ACTIVE_STATES))
            ).first()
            return ingestion_job_to_dict(record) if record else None

    def claim_next(
        self, worker_id: str, stale_timeout_s: int = 120
    ) -> dict[str, Any] | None:
        """Claim the oldest queued job so one worker owns it."""
        self.recover_stale(stale_timeout_s)
        with self._session_factory() as session, session.begin():
            candidate = session.scalars(
                select(IngestionJob)
                .where(IngestionJob.state == "queued")
                .order_by(IngestionJob.created_at.asc())
                .with_for_update(skip_locked=True)
            ).first()
            if candidate is None:
                return None
            now = _utcnow()
            candidate.state = "running"
            candidate.stage = "parsing"
            candidate.progress = 5
            candidate.worker_id = worker_id
            candidate.started_at = now
            candidate.heartbeat_at = now
            session.flush()
            return ingestion_job_to_dict(candidate)

    def heartbeat(
        self,
        job_id: str,
        worker_id: str,
        stage: str | None = None,
        progress: int | None = None,
    ) -> dict[str, Any] | None:
        """Refresh a running job owned by one worker."""
        with self._session_factory() as session, session.begin():
            record = session.get(IngestionJob, job_id)
            if (
                record is None
                or record.state != "running"
                or record.worker_id != worker_id
            ):
                return None
            record.heartbeat_at = _utcnow()
            if stage is not None:
                record.stage = stage
            if progress is not None:
                record.progress = progress
            session.flush()
            return ingestion_job_to_dict(record)

    def mark_ready(self, job_id: str, worker_id: str) -> dict[str, Any] | None:
        """Finish a job only while its worker still owns the claim."""
        with self._session_factory() as session, session.begin():
            record = session.get(IngestionJob, job_id)
            if not self._owned(record, worker_id):
                return None
            assert record is not None
            now = _utcnow()
            record.state = "ready"
            record.stage = "ready"
            record.progress = 100
            record.error_category = None
            record.error_message = None
            record.finished_at = now
            record.heartbeat_at = now
            session.flush()
            return ingestion_job_to_dict(record)

    def mark_failed(
        self, job_id: str, worker_id: str, category: str, message: str
    ) -> dict[str, Any] | None:
        """Fail a job only while its worker still owns the claim."""
        with self._session_factory() as session, session.begin():
            record = session.get(IngestionJob, job_id)
            if not self._owned(record, worker_id):
                return None
            assert record is not None
            now = _utcnow()
            record.state = "failed"
            record.stage = "failed"
            record.error_category = category[:64]
            record.error_message = message[:2000]
            record.finished_at = now
            record.heartbeat_at = now
            session.flush()
            return ingestion_job_to_dict(record)

    def mark_superseded(
        self, job_id: str, worker_id: str, category: str, message: str
    ) -> dict[str, Any] | None:
        """Stop a running job whose document entered the deleting state."""
        with self._session_factory() as session, session.begin():
            record = session.get(IngestionJob, job_id)
            if not self._owned(record, worker_id):
                return None
            assert record is not None
            now = _utcnow()
            record.state = "stale"
            record.stage = "stale"
            record.error_category = category[:64]
            record.error_message = message[:2000]
            record.finished_at = now
            record.heartbeat_at = now
            record.worker_id = None
            session.flush()
            return ingestion_job_to_dict(record)

    @staticmethod
    def _owned(record: IngestionJob | None, worker_id: str) -> bool:
        """Return whether this worker still holds the running claim."""
        return (
            record is not None
            and record.state == "running"
            and record.worker_id == worker_id
        )

    def recover_stale(self, stale_timeout_s: int = 120) -> list[str]:
        """Requeue running jobs whose worker stopped heartbeating."""
        recovered: list[str] = []
        with self._session_factory() as session, session.begin():
            running = session.scalars(
                select(IngestionJob).where(IngestionJob.state == "running")
            ).all()
            for record in running:
                age = _age_s(record.heartbeat_at or record.started_at)
                if age is not None and age >= stale_timeout_s:
                    record.state = "queued"
                    record.stage = "queued"
                    record.progress = 0
                    record.worker_id = None
                    record.started_at = None
                    record.heartbeat_at = None
                    recovered.append(record.id)
            session.flush()
        return recovered

    def count_active(self) -> int:
        """Return how many jobs are queued or running."""
        with self._session_factory() as session:
            count = session.scalar(
                select(func.count())
                .select_from(IngestionJob)
                .where(IngestionJob.state.in_(ACTIVE_STATES))
            )
            return int(count or 0)

    def cancel_active(self, file_id: str) -> int:
        """Supersede active jobs so deletion is not followed by late indexing."""
        with self._session_factory() as session, session.begin():
            active = session.scalars(
                select(IngestionJob)
                .where(IngestionJob.file_id == file_id)
                .where(IngestionJob.state.in_(ACTIVE_STATES))
            ).all()
            now = _utcnow()
            for record in active:
                record.state = "stale"
                record.stage = "stale"
                record.worker_id = None
                record.finished_at = now
            return len(active)

    def delete_for_file(self, file_id: str) -> None:
        """Remove every job for a deleted document."""
        with self._session_factory() as session, session.begin():
            records = session.scalars(
                select(IngestionJob).where(IngestionJob.file_id == file_id)
            ).all()
            for record in records:
                session.delete(record)
