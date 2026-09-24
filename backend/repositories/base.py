"""Shared repository session access and record conversion."""

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from db import get_session_factory


def to_record_dict(record: Any, **extra: Any) -> dict[str, Any]:
    """Convert a persisted record and its fields to a plain dictionary."""
    created = record.created_at or datetime.now(timezone.utc)
    data = {"id": record.id, "created_at": created.isoformat()}
    data.update(extra)
    return data


class BaseRepository:
    """Resolve one shared SQLAlchemy session factory."""

    def __init__(self, session_factory: Callable[[], Session] | None = None) -> None:
        """Bind the repository to an optional session factory."""
        self._factory_override = session_factory
        self._resolved: Callable[[], Session] | None = None

    @property
    def _session_factory(self) -> Callable[[], Session]:
        if self._factory_override is not None:
            return self._factory_override
        if self._resolved is None:
            self._resolved = get_session_factory()
        return self._resolved
