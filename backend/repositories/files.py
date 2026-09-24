"""Persistence for stored document metadata."""

from typing import Any

from sqlalchemy import select

from db import FileRecord
from repositories.base import BaseRepository, to_record_dict


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

    def delete_file(self, file_id: str) -> None:
        """Delete metadata for one document."""
        with self._session_factory() as session, session.begin():
            record = session.get(FileRecord, file_id)
            if record:
                session.delete(record)

    @staticmethod
    def _to_dict(record: FileRecord) -> dict[str, Any]:
        return to_record_dict(
            record,
            filename=record.filename,
            title=record.title,
            original_filename=record.original_filename,
            is_processed=record.is_processed,
        )
