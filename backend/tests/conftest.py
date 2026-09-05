"""
Shared test fixtures.

Convention: tests never talk to real Pinecone, LLM, Postgres, or any file
storage outside tmp dirs. Dummy environment variables are set before any
config import so module loading stays offline; anything that would hit a
network or disk is faked per-test with monkeypatch.
"""

import os

import pytest

DUMMY_ENV = {
    "DATABASE_URL": "sqlite:///:memory:",
    "PINECONE_API_KEY": "dummy-pinecone-key",
    "MODE": "google",
    "GOOGLE_API_KEY": "dummy-google-key",
    "GROQ_API_KEY": "dummy-groq-key",
    # Demo on by default so flow tests exercise business logic; auth tests
    # override DEMO_MODE explicitly.
    "DEMO_MODE": "true",
}

# Test modules import app (and trigger config validation) during collection,
# before any fixture runs, so the dummy environment must be in place at
# conftest import time too. The autouse fixture below re-applies it per test,
# since monkeypatch may have reverted individual variables mid-run.
os.environ.update(DUMMY_ENV)


@pytest.fixture(autouse=True)
def dummy_env(monkeypatch):
    """Provide a complete, offline-safe environment for every test."""
    # A developer's real .env must not leak into tests: config would
    # re-read it on reload and the dummy values below would be ignored.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    for key, value in DUMMY_ENV.items():
        monkeypatch.setenv(key, value)
    return DUMMY_ENV
