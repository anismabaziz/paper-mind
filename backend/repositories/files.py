"""Persistence for stored document metadata."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from db import FileRecord, IngestionJob
from repositories.base import BaseRepository, to_record_dict
from repositories.ingestion_jobs import ingestion_job_to_dict


class FileRepository(BaseRepository):
    """Read and update stored document metadata."""

    def create_file(
        self,
        filename: str,
        title: str | None = None,
        original_filename: str | None = None,
    ) -> dict[str, Any]:
        """Create metadata for one stored document."""
        with self._session_factory() as session, session.begin():
            record = FileRecord(
                filename=filename,
                title=title,
                original_filename=original_filename,
            )
            session.add(record)
            session.flush()
            return self._to_dict(record)

    def create_file_with_job(
        self,
        filename: str,
        title: str | None = None,
        original_filename: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Create the document and its first ingestion job in one commit."""
        with self._session_factory() as session, session.begin():
            record = FileRecord(
                filename=filename,
                title=title,
                original_filename=original_filename,
            )
            session.add(record)
            session.flush()
            job = IngestionJob(
                file_id=record.id,
                filename=filename,
                generation=1,
                state="queued",
                stage="queued",
                progress=0,
                attempt=1,
            )
            session.add(job)
            session.flush()
            return self._to_dict(record), ingestion_job_to_dict(job)

    def set_file_title(self, filename: str, title: str) -> None:
        """Replace the derived title for a stored document."""
        with self._session_factory() as session, session.begin():
            record = session.scalars(
                select(FileRecord).where(FileRecord.filename == filename)
            ).first()
            if record:
                record.title = title

    def get_file(self, filename: str) -> dict[str, Any] | None:
        """Return one document by its stable storage filename."""
        with self._session_factory() as session:
            record = session.scalars(
                select(FileRecord).where(FileRecord.filename == filename)
            ).first()
            return self._to_dict(record) if record else None

    def list_files(self) -> list[dict[str, Any]]:
        """Return document metadata in creation order."""
        with self._session_factory() as session:
            records = session.scalars(
                select(FileRecord).order_by(FileRecord.created_at)
            ).all()
            return [self._to_dict(record) for record in records]

    def set_processed(self, filename: str, status: bool = True) -> None:
        """Set whether a document's vectors and conversation are ready."""
        with self._session_factory() as session, session.begin():
            record = session.scalars(
                select(FileRecord).where(FileRecord.filename == filename)
            ).first()
            if record:
                record.is_processed = status

    def touch_opened(self, filename: str) -> dict[str, Any] | None:
        """Mark a document as opened now; drives the recent-readings order."""
        with self._session_factory() as session, session.begin():
            record = session.scalars(
                select(FileRecord).where(FileRecord.filename == filename)
            ).first()
            if not record:
                return None
            record.last_opened_at = datetime.now(timezone.utc)
            session.flush()
            return self._to_dict(record)

    def mark_deleting(self, filename: str) -> dict[str, Any] | None:
        """Enter the durable deleting state before external cleanup begins."""
        with self._session_factory() as session, session.begin():
            record = session.scalars(
                select(FileRecord).where(FileRecord.filename == filename)
            ).first()
            if not record:
                return None
            record.deletion_state = "deleting"
            record.deletion_error = None
            record.deletion_attempts = (record.deletion_attempts or 0) + 1
            session.flush()
            return self._to_dict(record)

    def mark_delete_failed(self, filename: str, error: str) -> None:
        """Retain retryable cleanup state after a partial deletion failure."""
        with self._session_factory() as session, session.begin():
            record = session.scalars(
                select(FileRecord).where(FileRecord.filename == filename)
            ).first()
            if record:
                record.deletion_state = "delete_failed"
                record.deletion_error = error[:2000]

    def delete_file(self, file_id: str) -> None:
        """Delete metadata for one document."""
        with self._session_factory() as session, session.begin():
            record = session.get(FileRecord, file_id)
            if record:
                session.delete(record)

    @staticmethod
    def _to_dict(record: FileRecord) -> dict[str, Any]:
        opened = record.last_opened_at
        return to_record_dict(
            record,
            filename=record.filename,
            title=record.title,
            original_filename=record.original_filename,
            is_processed=record.is_processed,
            index_generation=getattr(record, "index_generation", None),
            last_opened_at=opened.isoformat() if opened else None,
            deletion_state=getattr(record, "deletion_state", "active") or "active",
            deletion_error=getattr(record, "deletion_error", None),
            deletion_attempts=getattr(record, "deletion_attempts", 0) or 0,
        )
