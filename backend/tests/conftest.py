"""Shared fixtures for backend tests.

Plain ``pytest`` only collects ``tests/unit`` (see ``testpaths`` in
pyproject.toml). Those tests are fast: no network, no database, no Qdrant,
no model weights. Heavier suites live elsewhere and never run by accident:

- ``pytest tests/integration`` needs Flask test client, Postgres, or Qdrant.
- ``pytest tests/slow`` loads models or parses real PDFs.
"""

import copy
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from settings import Settings  # noqa: E402


@pytest.fixture()
def settings() -> Settings:
    """Return default settings with storage pointed at a temp dir."""
    return Settings()


@pytest.fixture()
def tmp_storage_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the process settings at an isolated storage directory."""
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    monkeypatch.setenv("STORAGE_DIR", str(storage_dir))
    return storage_dir


def make_file_record(**overrides):
    """Build a minimal document row dict for index-state tests."""
    record = {
        "id": "file-1",
        "filename": "abc123.pdf",
        "title": "Some paper",
        "is_processed": False,
        "index_generation": 0,
        "index_manifest": None,
        "index_stale_reason": None,
        "deletion_state": "active",
    }
    record.update(overrides)
    return copy.deepcopy(record)


def make_ingestion_job(state="queued", **overrides):
    """Build a minimal ingestion job dict for readiness tests."""
    job = {"id": "job-1", "state": state}
    job.update(overrides)
    return job
