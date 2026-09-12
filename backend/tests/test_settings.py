"""Tests for the typed Settings object and lazy provider singletons."""

from pathlib import Path

import pytest

import providers
import settings as settings_module
from settings import (
    Settings,
    app_secret_warning,
    get_settings,
    validate,
)


@pytest.fixture(autouse=True)
def fresh_settings_cache():
    """Each test builds its own Settings instead of reusing a memoized one."""
    settings_module.set_settings(None)
    yield
    settings_module.set_settings(None)


@pytest.fixture(autouse=True)
def fresh_providers():
    """Each test starts from unmemoized providers."""
    providers.reset_providers()
    yield
    providers.reset_providers()


def _clear(*names, monkeypatch=None):
    for name in names:
        monkeypatch.delenv(name, raising=False)


def test_defaults_cover_every_group(monkeypatch):
    """Do test defaults cover every group."""
    _clear(
        "DATABASE_URL",
        "STORAGE_DIR",
        "QDRANT_URL",
        "LOCAL_EMBEDDING_MODEL",
        "CHUNK_SIZE_TOKENS",
        "CHUNK_OVERLAP_TOKENS",
        "RERANK_MODEL",
        "RERANK",
        "USE_DOCLING",
        "APP_SECRET",
        "JWT_SECRET",
        monkeypatch=monkeypatch,
    )
    monkeypatch.setenv("DATABASE_URL", "postgresql://papermind")

    s = Settings()

    assert s.database.database_url == "postgresql://papermind"
    assert s.storage.storage_dir == settings_module.BACKEND_DIR / "data" / "storage"
    assert s.vector.qdrant_url == "http://localhost:6333"
    assert s.vector.index_name == "pdf-index"
    assert s.embedding.embedding_model == "BAAI/bge-m3"
    assert s.chunking.chunk_size_tokens == 512
    assert s.chunking.chunk_overlap_tokens == 50
    assert s.rerank.rerank_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert s.rerank.enabled is False
    assert s.parsing.use_docling == "auto"
    assert s.auth.app_secret is None


def test_env_overrides_bind_to_groups(monkeypatch):
    """Do test env overrides bind to groups."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://elsewhere")
    monkeypatch.setenv("STORAGE_DIR", "/tmp/papermind-uploads")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("CHUNK_SIZE_TOKENS", "256")
    monkeypatch.setenv("RERANK", "true")
    monkeypatch.setenv("APP_SECRET", "s3cret")

    s = Settings()

    assert s.storage.storage_dir == Path("/tmp/papermind-uploads")
    assert s.vector.qdrant_url == "http://qdrant:6333"
    assert s.chunking.chunk_size_tokens == 256
    assert s.rerank.enabled is True
    assert s.auth.app_secret == "s3cret"


def test_jwt_secret_alias_still_maps_to_app_secret(monkeypatch):
    """Legacy JWT_SECRET env var still populates APP_SECRET."""
    _clear("APP_SECRET", monkeypatch=monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgresql://papermind")
    monkeypatch.setenv("JWT_SECRET", "legacy-secret")

    s = Settings()
    assert s.auth.app_secret == "legacy-secret"


def test_building_settings_never_raises_without_env(monkeypatch):
    """Imports stay side-effect free: an empty env yields an empty URL."""
    _clear("DATABASE_URL", monkeypatch=monkeypatch)

    assert Settings().database.database_url == ""


def test_empty_database_url_is_treated_as_missing(monkeypatch):
    """Do test empty database url is treated as missing."""
    monkeypatch.setenv("DATABASE_URL", "")

    with pytest.raises(SystemExit):
        validate()


def test_validate_exits_with_named_variable_in_message(capsys, monkeypatch):
    """Do test validate exits with named variable in message."""
    _clear("DATABASE_URL", monkeypatch=monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        validate()

    assert excinfo.value.code == 1
    stderr = capsys.readouterr().err
    assert "DATABASE_URL" in stderr
    assert ".env.example" in stderr
    assert "Traceback" not in stderr


def test_validate_warns_when_app_secret_unset(capsys, monkeypatch):
    """Do test validate warns when app secret unset."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://papermind")
    _clear("APP_SECRET", "JWT_SECRET", monkeypatch=monkeypatch)

    validate()

    stderr = capsys.readouterr().err
    assert "APP_SECRET" in stderr


def test_validate_is_quiet_when_app_secret_set(capsys, monkeypatch):
    """Do test validate is quiet when app secret set."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://papermind")
    monkeypatch.setenv("APP_SECRET", "s3cret")

    validate()

    assert capsys.readouterr().err == ""


def test_app_secret_warning_helper(monkeypatch):
    """Do test app secret warning helper."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://papermind")

    _clear("APP_SECRET", "JWT_SECRET", monkeypatch=monkeypatch)
    assert "APP_SECRET" in app_secret_warning(Settings())

    monkeypatch.setenv("APP_SECRET", "s3cret")
    assert app_secret_warning(Settings()) is None


def test_accessor_returns_process_wide_instance(monkeypatch):
    """Do test accessor returns process wide instance."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://one")

    first = get_settings()
    second = get_settings()

    assert first is second


def test_accessor_cache_clear_builds_fresh_settings(monkeypatch):
    """Do test accessor cache clear builds fresh settings."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://one")
    assert get_settings().database.database_url == "postgresql://one"

    monkeypatch.setenv("DATABASE_URL", "postgresql://two")
    settings_module.set_settings(None)
    assert get_settings().database.database_url == "postgresql://two"


def test_booting_without_env_exits_readably(capsys, monkeypatch):
    """
    End to end: composing the app with an empty env exits with the named.

        variable on stderr, not a library traceback.
    """
    import pathlib
    import subprocess
    import sys

    backend_dir = pathlib.Path(__file__).resolve().parent.parent
    # Empty-string values shadow any local .env (dotenv does not override
    # existing vars) and count as missing to the validator.
    # Provider keys are app-settings-backed now; only DATABASE_URL is required.
    result = subprocess.run(
        [sys.executable, "-c", "from app import create_app; create_app()"],
        capture_output=True,
        text=True,
        cwd=backend_dir,
        env={"DATABASE_URL": ""},
    )
    assert result.returncode == 1
    assert "DATABASE_URL" in result.stderr
    assert "Traceback" not in result.stderr


def test_providers_stay_lazy_until_first_use(monkeypatch):
    """Do test providers stay lazy until first use."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://papermind")
    constructed = []

    class FakeQdrantClient:
        def __init__(self, url):
            constructed.append(url)

    monkeypatch.setattr("qdrant_client.QdrantClient", FakeQdrantClient)

    # Importing providers and reading accessors must not build anything.
    assert constructed == []

    providers.get_qdrant_client()
    assert constructed == ["http://localhost:6333"]


def test_provider_client_is_built_once_from_settings(monkeypatch):
    """Do test provider client is built once from settings."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://papermind")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")

    built = []

    class FakeQdrantClient:
        def __init__(self, url):
            self.url = url
            built.append(self)

    monkeypatch.setattr("qdrant_client.QdrantClient", FakeQdrantClient)

    first = providers.get_qdrant_client()
    second = providers.get_qdrant_client()

    assert built == [first]
    assert first.url == "http://qdrant:6333"
    assert first is second


def test_provider_index_is_built_once_from_settings(monkeypatch):
    """Do test provider index is built once from settings."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://papermind")

    built = []

    class FakeIndexAdapter:
        def __init__(self, client, index_name):
            self.client = client
            self.index_name = index_name
            built.append(self)

    monkeypatch.setattr(
        "services.retrieval.qdrant_store.QdrantIndexAdapter", FakeIndexAdapter
    )

    first = providers.get_vector_index()
    second = providers.get_vector_index()

    assert built == [first]
    assert first.index_name == "pdf-index"
    assert first is second
