"""
Shared test fixtures.

The default suite uses fake Qdrant, chat, and storage adapters with in-memory
SQLite. The separately marked full-stack suite explicitly uses real Postgres
and Qdrant with deterministic model adapters. A pinned Settings instance keeps
developer environment values out of both suites.
"""

from pathlib import Path

# Settings loads backend/.env when the app boots; tests must not inherit a
# developer's values, so dotenv is neutralized before any module imports it.
import dotenv

dotenv.load_dotenv = lambda *a, **k: False

import pytest

import settings as settings_module
from settings import (
    AuthSettings,
    ChunkingSettings,
    DatabaseSettings,
    EmbeddingSettings,
    ParsingSettings,
    RerankSettings,
    Settings,
    StorageSettings,
    VectorSettings,
)

# Pinned offline-safe settings: sqlite in memory, deterministic secret.
# Every group is explicit so a real .env cannot leak through.
TEST_SETTINGS = Settings(
    database=DatabaseSettings(database_url="sqlite:///:memory:"),
    storage=StorageSettings(
        storage_dir=settings_module.BACKEND_DIR / "data" / "storage"
    ),
    vector=VectorSettings(),
    embedding=EmbeddingSettings(),
    chunking=ChunkingSettings(),
    rerank=RerankSettings(),
    parsing=ParsingSettings(),
    auth=AuthSettings(app_secret="test-app-secret"),
)


@pytest.fixture(autouse=True)
def test_settings():
    """Install the pinned test Settings for the duration of each test."""
    settings_module.set_settings(TEST_SETTINGS)
    yield TEST_SETTINGS
    settings_module.set_settings(None)


@pytest.fixture
def settings_obj():
    """Return the installed test Settings; tweak group fields via monkeypatch."""
    return settings_module.get_settings()
