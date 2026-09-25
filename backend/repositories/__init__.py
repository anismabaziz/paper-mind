"""Aggregate-specific persistence interfaces."""

from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy.orm import Session

from db import get_session_factory
from repositories.app_settings import AppSettingsRepository
from repositories.conversations import ConversationRepository
from repositories.files import FileRepository
from repositories.index_cleanups import IndexCleanupRepository
from repositories.ingestion_jobs import IngestionJobRepository


@dataclass(frozen=True)
class Repositories:
    """Persistence interfaces grouped by aggregate."""

    files: FileRepository
    app_settings: AppSettingsRepository
    conversations: ConversationRepository
    ingestion_jobs: IngestionJobRepository
    index_cleanups: IndexCleanupRepository


def build_repositories(
    session_factory: Callable[[], Session] | None = None,
) -> Repositories:
    """Build every repository over one shared session factory."""
    factory = session_factory or get_session_factory()
    return Repositories(
        files=FileRepository(factory),
        app_settings=AppSettingsRepository(factory),
        conversations=ConversationRepository(factory),
        ingestion_jobs=IngestionJobRepository(factory),
        index_cleanups=IndexCleanupRepository(factory),
    )


__all__ = [
    "AppSettingsRepository",
    "ConversationRepository",
    "FileRepository",
    "IngestionJobRepository",
    "IndexCleanupRepository",
    "Repositories",
    "build_repositories",
]
