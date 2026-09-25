"""
Durable cleanup of superseded index generations.

Deleting the superseded vectors cannot live in the worker alone: a restart
between activation and deletion would leak the old generation forever. Each
replacement is therefore written down before it is activated, and the removal
is retried from the recorded work until it succeeds.
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from db import IndexGenerationCleanup
from repositories.base import BaseRepository, to_record_dict

# Vectors written before generations were recorded carry no generation, so
# they are cleaned up as generation zero.
UNVERSIONED_GENERATION = 0


def generation_key(generation: int | None) -> int:
    """Return the stored key for a generation, numbering unversioned as zero."""
    return UNVERSIONED_GENERATION if generation is None else generation


def _utcnow() -> datetime:
    """Return the current UTC time."""
    return datetime.now(timezone.utc)


def _to_dict(record: IndexGenerationCleanup) -> dict[str, Any]:
    finished = record.finished_at
    return to_record_dict(
        record,
        file_id=record.file_id,
        filename=record.filename,
        generation=record.generation,
        state=record.state,
        attempts=record.attempts,
        error=record.error,
        finished_at=finished.isoformat() if finished else None,
    )


class IndexCleanupRepository(BaseRepository):
    """Track the removals a Document's generation changes still owe."""

    def schedule(
        self, file_id: str, filename: str, generation: int | None
    ) -> dict[str, Any]:
        """Record that one generation is superseded, without duplicating work."""
        target = generation_key(generation)
        with self._session_factory() as session, session.begin():
            existing = session.scalars(
                select(IndexGenerationCleanup)
                .where(IndexGenerationCleanup.file_id == file_id)
                .where(IndexGenerationCleanup.generation == target)
                .with_for_update()
            ).first()
            if existing is not None:
                existing.state = "pending"
                existing.error = None
                existing.finished_at = None
                session.flush()
                return _to_dict(existing)
            record = IndexGenerationCleanup(
                file_id=file_id,
                filename=filename,
                generation=target,
                state="pending",
            )
            session.add(record)
            try:
                session.flush()
            except IntegrityError:
                # Another writer recorded the same removal first.
                session.rollback()
                return self.schedule(file_id, filename, generation)
            return _to_dict(record)

    def list_pending(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the removals still owed, oldest first."""
        with self._session_factory() as session:
            records = session.scalars(
                select(IndexGenerationCleanup)
                .where(IndexGenerationCleanup.state == "pending")
                .order_by(IndexGenerationCleanup.created_at)
                .limit(limit)
            ).all()
            return [_to_dict(record) for record in records]

    def mark_done(self, cleanup_id: str) -> None:
        """Finish a removal so it is not attempted again."""
        with self._session_factory() as session, session.begin():
            record = session.get(
                IndexGenerationCleanup, cleanup_id, with_for_update=True
            )
            if record is None:
                return
            record.state = "done"
            record.error = None
            record.finished_at = _utcnow()

    def record_failure(self, cleanup_id: str, error: str) -> None:
        """Keep a failed removal pending so the next run retries it."""
        with self._session_factory() as session, session.begin():
            record = session.get(
                IndexGenerationCleanup, cleanup_id, with_for_update=True
            )
            if record is None:
                return
            record.attempts = (record.attempts or 0) + 1
            record.error = error[:2000]
            record.finished_at = None

    def delete_for_file(self, file_id: str) -> None:
        """Drop the recorded removals of a document that no longer exists."""
        with self._session_factory() as session, session.begin():
            records = session.scalars(
                select(IndexGenerationCleanup).where(
                    IndexGenerationCleanup.file_id == file_id
                )
            ).all()
            for record in records:
                session.delete(record)
