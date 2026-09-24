"""Aggregate-specific persistence interfaces."""

from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy.orm import Session

from db import get_session_factory
from repositories.app_settings import AppSettingsRepository
from repositories.conversations import ConversationRepository
from repositories.files import FileRepository
from repositories.ingestion_jobs import IngestionJobRepository


@dataclass(frozen=True)
class Repositories:
    """Persistence interfaces grouped by aggregate."""

    files: FileRepository
    app_settings: AppSettingsRepository
    conversations: ConversationRepository
    ingestion_jobs: IngestionJobRepository


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
    )


__all__ = [
    "AppSettingsRepository",
    "ConversationRepository",
    "FileRepository",
    "IngestionJobRepository",
    "Repositories",
    "build_repositories",
]
