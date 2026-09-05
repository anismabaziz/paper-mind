"""Tests for the typed Settings object and lazy provider singletons."""

from pathlib import Path

import pytest

import providers
import settings as settings_module
from pydantic import ValidationError
from settings import (
    Settings,
    get_settings,
    jwt_secret_warning,
    validate,
)


@pytest.fixture(autouse=True)
def fresh_settings_cache():
    """Each test builds its own Settings instead of reusing a memoized one."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


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
        "JWT_SECRET",
        "DEMO_MODE",
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
    assert s.auth.jwt_secret is None
    assert s.auth.demo_mode is False


def test_env_overrides_bind_to_groups(monkeypatch):
    """Do test env overrides bind to groups."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://elsewhere")
    monkeypatch.setenv("STORAGE_DIR", "/tmp/papermind-uploads")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("CHUNK_SIZE_TOKENS", "256")
    monkeypatch.setenv("RERANK", "true")
    monkeypatch.setenv("JWT_SECRET", "s3cret")
    monkeypatch.setenv("DEMO_MODE", "TRUE")

    s = Settings()

    assert s.storage.storage_dir == Path("/tmp/papermind-uploads")
    assert s.vector.qdrant_url == "http://qdrant:6333"
    assert s.chunking.chunk_size_tokens == 256
    assert s.rerank.enabled is True
    assert s.auth.jwt_secret == "s3cret"
    assert s.auth.demo_mode is True


def test_missing_database_url_is_named(monkeypatch):
    """Do test missing database url is named."""
    _clear("DATABASE_URL", monkeypatch=monkeypatch)

    with pytest.raises(ValidationError) as excinfo:
        Settings()

    message = str(excinfo.value)
    assert "DATABASE_URL" in message
    assert "Field required" in message


def test_empty_database_url_is_treated_as_missing(monkeypatch):
    """Do test empty database url is treated as missing."""
    monkeypatch.setenv("DATABASE_URL", "")

    with pytest.raises(ValidationError):
        Settings()


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


def test_validate_warns_when_jwt_secret_unset_outside_demo(capsys, monkeypatch):
    """Do test validate warns when jwt secret unset outside demo."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://papermind")
    monkeypatch.setenv("DEMO_MODE", "false")
    _clear("JWT_SECRET", monkeypatch=monkeypatch)

    validate()

    stderr = capsys.readouterr().err
    assert "JWT_SECRET is unset" in stderr


def test_validate_is_quiet_in_demo_mode_without_secret(capsys, monkeypatch):
    """Do test validate is quiet in demo mode without secret."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://papermind")
    monkeypatch.setenv("DEMO_MODE", "true")
    _clear("JWT_SECRET", monkeypatch=monkeypatch)

    validate()

    assert capsys.readouterr().err == ""


def test_jwt_secret_warning_helper(monkeypatch):
    """Do test jwt secret warning helper."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://papermind")

    monkeypatch.setenv("DEMO_MODE", "false")
    _clear("JWT_SECRET", monkeypatch=monkeypatch)
    assert "JWT_SECRET" in jwt_secret_warning(Settings())

    monkeypatch.setenv("JWT_SECRET", "s3cret")
    assert jwt_secret_warning(Settings()) is None

    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.setenv("DEMO_MODE", "true")
    assert jwt_secret_warning(Settings()) is None


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
    get_settings.cache_clear()
    assert get_settings().database.database_url == "postgresql://two"


def test_providers_stay_lazy_until_first_use(monkeypatch):
    """Do test providers stay lazy until first use."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://papermind")

    assert providers._qdrant_client is None
    assert providers._qdrant_index is None


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


def test_provider_index_honors_installed_fake_and_reset():
    """Do test provider index honors installed fake and reset."""
    fake = object()
    providers._qdrant_index = fake
    assert providers.get_vector_index() is fake

    providers.reset_providers()
    assert providers._qdrant_index is None
    assert providers._qdrant_client is None
