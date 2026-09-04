import importlib
import subprocess
import sys

import config


def reload_config():
    return importlib.reload(config)


def test_all_required_vars_missing_are_named(monkeypatch):
    # Pinecone key is only required when VECTOR_BACKEND=pinecone; default is qdrant
    monkeypatch.setenv("VECTOR_BACKEND", "pinecone")
    for var in (
        "DATABASE_URL",
        "PINECONE_API_KEY",
        "MODE",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    cfg = reload_config()
    missing = cfg.missing_required_vars()

    assert "DATABASE_URL" in missing
    assert "PINECONE_API_KEY" in missing
    # MODE defaults to "google", so the Google key is the one required
    assert "GOOGLE_API_KEY" in missing

    # With qdrant (default), Pinecone key is not required
    monkeypatch.setenv("VECTOR_BACKEND", "qdrant")
    # Keep PINECONE missing
    cfg = reload_config()
    missing = cfg.missing_required_vars()
    assert "PINECONE_API_KEY" not in missing
    assert "DATABASE_URL" in missing


def test_provider_key_follows_mode(monkeypatch):
    monkeypatch.setenv("MODE", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    cfg = reload_config()
    assert "GROQ_API_KEY" in cfg.missing_required_vars()
    assert "GOOGLE_API_KEY" not in cfg.missing_required_vars()


def test_invalid_mode_is_reported(monkeypatch):
    monkeypatch.setenv("MODE", "azure")

    cfg = reload_config()
    assert any("MODE" in var for var in cfg.missing_required_vars())


def test_complete_env_validates_clean():
    cfg = reload_config()
    assert cfg.missing_required_vars() == []


def test_validate_exits_with_named_variable_in_message(capsys, monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "pinecone")
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)

    cfg = reload_config()
    try:
        cfg.validate()
    except SystemExit as e:
        assert e.code == 1
    else:
        raise AssertionError("validate() should exit when variables are missing")

    stderr = capsys.readouterr().err
    assert "PINECONE_API_KEY" in stderr
    assert ".env.example" in stderr


def test_validate_passes_with_complete_env(capsys):
    reload_config().validate()
    assert capsys.readouterr().err == ""


def test_import_does_not_build_clients(monkeypatch):
    cfg = reload_config()
    assert cfg._pinecone_index is None
    assert cfg._qdrant_index is None
    assert cfg._qdrant_client is None
    assert cfg._genai_client is None
    assert cfg._groq_client is None


def test_vector_backend_defaults_to_qdrant(monkeypatch):
    monkeypatch.delenv("VECTOR_BACKEND", raising=False)
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    cfg = reload_config()
    assert cfg._vector_backend() == "qdrant"
    # qdrant default does not require Pinecone key
    assert "PINECONE_API_KEY" not in cfg.missing_required_vars()


def test_qdrant_backend_does_not_require_pinecone(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "qdrant")
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    cfg = reload_config()
    assert "PINECONE_API_KEY" not in cfg.missing_required_vars()


def test_invalid_vector_backend_is_reported(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "weaviate")
    cfg = reload_config()
    assert any("VECTOR_BACKEND" in var for var in cfg.missing_required_vars())


def test_vector_backend_switch_uses_correct_index(monkeypatch):
    """
    Both backends are exercised through fakes so the switch never hits the network.
    """
    class FakePinecone:
        def upsert(self, vectors): return {"upserted": len(vectors)}
        def query(self, **kwargs): return {"matches": []}
        def delete(self, **kwargs): return {}

    class FakeQdrant:
        def upsert(self, vectors): return {"upserted": len(vectors)}
        def query(self, **kwargs): return {"matches": []}
        def delete(self, **kwargs): return {}

    fake_pinecone = FakePinecone()
    fake_qdrant = FakeQdrant()

    monkeypatch.setenv("VECTOR_BACKEND", "pinecone")
    monkeypatch.setattr(config, "_pinecone_index", fake_pinecone)
    monkeypatch.setattr(config, "_qdrant_index", None)
    cfg = reload_config()
    # Re-apply fake after reload (reload clears the module globals)
    monkeypatch.setattr(cfg, "_pinecone_index", fake_pinecone)
    monkeypatch.setattr(cfg, "_qdrant_index", None)
    assert cfg.get_vector_index() is fake_pinecone

    monkeypatch.setenv("VECTOR_BACKEND", "qdrant")
    monkeypatch.setattr(cfg, "_qdrant_index", fake_qdrant)
    # Legacy Pinecone fake should not shadow the Qdrant fake when Qdrant is selected
    monkeypatch.setattr(cfg, "_pinecone_index", fake_pinecone)
    assert cfg.get_vector_index() is fake_qdrant


def test_booting_without_env_exits_readably():
    """
        End to end: starting the app with an empty env exits with the named
            variable on stderr, not a library traceback.
    """
    import pathlib

    backend_dir = pathlib.Path(__file__).resolve().parent.parent
    # Empty-string values shadow any local .env (dotenv does not override
    # existing vars) and count as missing to the validator.
    empty_env = {var: "" for var in (
        "DATABASE_URL",
        "PINECONE_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
    )}
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
