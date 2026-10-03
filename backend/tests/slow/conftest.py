"""Slow fixtures: real Postgres, real Qdrant, real model weights.

Run with ``pytest tests/slow``. Never collected by plain ``pytest``.
Infra suites skip when their service is unreachable, so the files also run
on machines without the compose stack. Model suites load the real weights
from the Hugging Face cache and fail loudly when the cache is missing,
because a model test that silently skipped would prove nothing.
"""

import os
import socket
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

TEST_QDRANT_URL = "http://localhost:6333"
TEST_COLLECTION = "papermind-test-slow"


def test_database_url():
    """Build the test database URL from the environment, never committed.

    Reads the same ``POSTGRES_USER`` / ``POSTGRES_PASSWORD`` the compose
    stack runs on (``backend/.env``, loaded through settings). CI sets both
    explicitly and points at its own service.
    """
    user = os.environ.get("POSTGRES_USER", "papermind")
    password = os.environ.get("POSTGRES_PASSWORD", "")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    name = os.environ.get("POSTGRES_TEST_DB", "papermind_test")
    return f"postgresql+psycopg://{user}:{password}@{host}:5432/{name}"


def _tcp_open(host, port, timeout=2.0):
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


def qdrant_reachable():
    """Return True when a Qdrant answers on the test URL."""
    try:
        import urllib.request

        with urllib.request.urlopen(TEST_QDRANT_URL + "/healthz", timeout=3) as reply:
            return reply.status == 200
    except Exception:
        return False


def postgres_reachable():
    """Return True when the test database accepts connections."""
    return _tcp_open("localhost", 5432)


@pytest.fixture(scope="session")
def slow_settings(tmp_path_factory):
    """Test settings over an isolated storage dir, installed globally."""
    import settings as settings_module

    try:
        previous = settings_module.get_settings()
    except Exception:
        previous = None
    candidate = settings_module.Settings()
    candidate.storage.storage_dir = tmp_path_factory.mktemp("slow-storage")
    candidate.database.database_url = test_database_url()
    candidate.auth.app_secret = "slow-test-secret"
    settings_module.set_settings(candidate)
    yield candidate
    settings_module.set_settings(previous)


@pytest.fixture(scope="session")
def pg_engine(slow_settings):
    """Engine over the isolated test database with tables created."""
    if not postgres_reachable():
        pytest.skip("postgres not reachable at localhost:5432")
    import db

    db._engine = None
    db.SessionLocal = None
    from sqlalchemy import create_engine

    url = test_database_url()
    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1")
    except Exception as exc:
        pytest.skip(f"cannot connect to test database: {exc}")
    db.Base.metadata.create_all(engine)
    yield engine
    db._engine = None
    db.SessionLocal = None


@pytest.fixture()
def repositories(pg_engine, slow_settings):
    """Real repositories over the test database, emptied after each test."""
    import db
    from repositories import build_repositories

    factory = pg_engine and __import__(
        "sqlalchemy.orm", fromlist=["sessionmaker"]
    ).sessionmaker(bind=pg_engine)
    repos = build_repositories(session_factory=factory)
    yield repos
    with pg_engine.begin() as connection:
        for table in (
            "sources",
            "turns",
            "conversations",
            "ingestion_jobs",
            "index_generation_cleanups",
            "app_settings",
            "files",
        ):
            connection.exec_driver_sql(f"DELETE FROM {table}")
    db._engine = None
    db.SessionLocal = None


@pytest.fixture(scope="session")
def qdrant_client():
    """Real Qdrant client, or a skip when the service is down."""
    if not qdrant_reachable():
        pytest.skip("qdrant not reachable at localhost:6333")
    from qdrant_client import QdrantClient

    client = QdrantClient(url=TEST_QDRANT_URL, timeout=30)
    yield client
    client.close()


@pytest.fixture()
def test_collection(qdrant_client):
    """An empty test collection, removed after the test."""
    if qdrant_client.collection_exists(TEST_COLLECTION):
        qdrant_client.delete_collection(TEST_COLLECTION)
    yield TEST_COLLECTION
    if qdrant_client.collection_exists(TEST_COLLECTION):
        qdrant_client.delete_collection(TEST_COLLECTION)


@pytest.fixture(scope="session")
def cached_models():
    """Prefer the local HF cache; download only when weights are missing.

    Offline when both weights are cached, so a local run never stalls on the
    network. Online on a fresh machine (like CI), where the download is the
    point.
    """
    hub = (
        Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    )
    wanted = ("models--BAAI--bge-m3", "models--cross-encoder--ms-marco-MiniLM-L-6-v2")
    if all((hub / name).exists() for name in wanted):
        os.environ["HF_HUB_OFFLINE"] = "1"
    yield


def make_pdf_bytes(pages):
    """Build a small multi-page PDF in memory. pages is a list of strings."""
    import io

    import pymupdf

    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.set_metadata({"title": "Slow Test Paper"})
    buffer = io.BytesIO()
    doc.save(buffer)
    doc.close()
    return buffer.getvalue()
