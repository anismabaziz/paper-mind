"""Module docstring."""

import importlib
import subprocess
import sys

import config


def reload_config():
    """Do reload config."""
    return importlib.reload(config)


def test_all_required_vars_missing_are_named(monkeypatch):
    """Do test all required vars missing are named."""
    monkeypatch.delenv("DATABASE_URL", raising=False)

    cfg = reload_config()
    missing = cfg.missing_required_vars()

    assert "DATABASE_URL" in missing


def test_complete_env_validates_clean():
    """Do test complete env validates clean."""
    cfg = reload_config()
    assert cfg.missing_required_vars() == []


def test_validate_exits_with_named_variable_in_message(capsys, monkeypatch):
    """Do test validate exits with named variable in message."""
    monkeypatch.delenv("DATABASE_URL", raising=False)

    cfg = reload_config()
    try:
        cfg.validate()
    except SystemExit as e:
        assert e.code == 1
    else:
        raise AssertionError("validate() should exit when variables are missing")

    stderr = capsys.readouterr().err
    assert "DATABASE_URL" in stderr
    assert ".env.example" in stderr


def test_validate_passes_with_complete_env(capsys):
    """Do test validate passes with complete env."""
    reload_config().validate()
    assert capsys.readouterr().err == ""


def test_import_does_not_build_clients(monkeypatch):
    """Do test import does not build clients."""
    cfg = reload_config()
    assert cfg._qdrant_index is None
    assert cfg._qdrant_client is None


def test_vector_backend_defaults_to_qdrant(monkeypatch):
    """Do test vector backend defaults to qdrant."""
    monkeypatch.delenv("VECTOR_BACKEND", raising=False)
    cfg = reload_config()
    assert cfg._vector_backend() == "qdrant"


def test_invalid_vector_backend_is_reported(monkeypatch):
    """Do test invalid vector backend is reported."""
    monkeypatch.setenv("VECTOR_BACKEND", "weaviate")
    cfg = reload_config()
    assert any("VECTOR_BACKEND" in var for var in cfg.missing_required_vars())


def test_vector_backend_switch_uses_correct_index(monkeypatch):
    """The Qdrant backend is exercised through a fake so the seam never hits the network."""

    class FakeQdrant:
        """FakeQdrant."""

        def upsert(self, vectors):
            """Do upsert."""
            return {"upserted": len(vectors)}

        def query(self, **kwargs):
            """Do query."""
            return {"matches": []}

        def delete(self, **kwargs):
            """Do delete."""
            return {}

    fake_qdrant = FakeQdrant()

    monkeypatch.setenv("VECTOR_BACKEND", "qdrant")
    monkeypatch.setattr(config, "_qdrant_index", fake_qdrant)
    assert config.get_vector_index() is fake_qdrant


def test_booting_without_env_exits_readably():
    """
    End to end: starting the app with an empty env exits with the named.

        variable on stderr, not a library traceback.
    """
    import pathlib

    backend_dir = pathlib.Path(__file__).resolve().parent.parent
    # Empty-string values shadow any local .env (dotenv does not override
    # existing vars) and count as missing to the validator.
    # Provider keys are per-user settings now; only DATABASE_URL is required.
    empty_env = {"DATABASE_URL": ""}
    result = subprocess.run(
        [sys.executable, "-c", "import app"],
        capture_output=True,
        text=True,
        cwd=backend_dir,
        env=empty_env,
    )
    assert result.returncode == 1
    assert "DATABASE_URL" in result.stderr
    assert "Traceback" not in result.stderr
