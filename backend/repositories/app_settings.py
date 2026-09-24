"""Persistence for the global application settings row."""

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from db import APP_SETTINGS_ID, AppSettings
from repositories.base import BaseRepository


class AppSettingsRepository(BaseRepository):
    """Read and update the single global application settings row."""

    @staticmethod
    def _to_dict(record: AppSettings) -> dict[str, Any]:
        return {
            "id": record.id,
            "provider": record.provider,
            "model": record.model,
            "encrypted_api_key": record.encrypted_api_key,
            "updated_at": (
                record.updated_at.isoformat() if record.updated_at else None
            ),
        }

    @staticmethod
    def _row(session: Session) -> AppSettings | None:
        record = session.get(AppSettings, APP_SETTINGS_ID)
        if record is None:
            record = session.scalars(select(AppSettings).limit(1)).first()
        return record

    def get_app_settings(self) -> dict[str, Any] | None:
        """Return the global settings row, or None when it has not been saved."""
        with self._session_factory() as session:
            record = self._row(session)
            return self._to_dict(record) if record else None

    def upsert_app_settings(
        self, provider: str, model: str, encrypted_api_key: str
    ) -> dict[str, Any]:
        """Create or replace the global settings row."""
        with self._session_factory() as session, session.begin():
            record = self._row(session)
            if record is None:
                record = AppSettings(id=APP_SETTINGS_ID)
                session.add(record)
            record.provider = provider
            record.model = model
            record.encrypted_api_key = encrypted_api_key
            session.flush()
            return self._to_dict(record)
