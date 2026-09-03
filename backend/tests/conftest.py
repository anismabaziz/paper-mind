"""Shared test fixtures.

Convention: tests never talk to real Pinecone, LLM, Postgres, or any file
storage outside tmp dirs. Dummy environment variables are set before any
config import so module loading stays offline; anything that would hit a
network or disk is faked per-test with monkeypatch.
"""

import pytest

DUMMY_ENV = {
    "DATABASE_URL": "sqlite:///:memory:",
    "PINECONE_API_KEY": "dummy-pinecone-key",
    "MODE": "google",
    "GOOGLE_API_KEY": "dummy-google-key",
    "GROQ_API_KEY": "dummy-groq-key",
    "DEMO_MODE": "false",
}


@pytest.fixture(autouse=True)
def dummy_env(monkeypatch):
    """Provide a complete, offline-safe environment for every test."""
    # A developer's real .env must not leak into tests: config would
    # re-read it on reload and the dummy values below would be ignored.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    for key, value in DUMMY_ENV.items():
        monkeypatch.setenv(key, value)
    return DUMMY_ENV
