"""
Settings API route tests.

Run through the Flask test client with the same fakes as the flow tests:
in-memory sqlite repository, no real provider calls (verification is
monkeypatched). Demo mode is on by default per conftest; the auth test
turns it off explicitly.
"""

import importlib

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db import Base, Repository, User
from services.secrets_service import decrypt_api_key


@pytest.fixture
def app_module(monkeypatch):
    """Do app module."""
    import app

    return importlib.import_module("app")


@pytest.fixture
def repo():
    """Do repo."""
    engine = create_engine(
        "sqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return Repository(sessionmaker(bind=engine))


@pytest.fixture
def demo_user(repo):
    """Seed the demo user the routes resolve to in demo mode."""
    with repo._session_factory() as session, session.begin():
        user = User(email="demo@papermind.local", password_hash="x")
        session.add(user)
        session.flush()
        return user.id


@pytest.fixture
def jwt_secret(monkeypatch):
    """Set a deterministic JWT_SECRET for crypto calls."""
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    return "test-secret"


@pytest.fixture
def client(app_module, repo, demo_user, monkeypatch, jwt_secret):
    """Do client."""
    monkeypatch.setattr(app_module, "repository", repo)
    with app_module.app.test_client() as client:
        yield client, app_module


def test_settings_require_auth(app_module, repo, monkeypatch, demo_user):
    """Do test settings require auth."""
    monkeypatch.setattr(app_module, "repository", repo)
    monkeypatch.setenv("DEMO_MODE", "false")
    with app_module.app.test_client() as client:
        for method, path in [
            (client.get, "/settings"),
            (client.put, "/settings"),
            (client.post, "/settings/verify"),
        ]:
            response = method(path)
            assert response.status_code == 401, path


def test_get_settings_empty_state(client):
    """Empty settings return the supported-models map and no provider."""
    client, _ = client
    response = client.get("/settings")
    assert response.status_code == 200
    body = response.get_json()
    assert body["provider"] is None
    assert body["model"] is None
    assert body["masked_key"] is None
    assert set(body["supported_models"]) == {"google", "groq"}


def test_put_rejects_unknown_provider(client):
    """Do test put rejects unknown provider."""
    client, _ = client
    response = client.put(
        "/settings",
        json={"provider": "openai", "model": "gpt-4o", "api_key": "sk-test"},
    )
    assert response.status_code == 400
    assert "Unsupported provider" in response.get_json()["error"]


def test_put_rejects_unknown_model(client):
    """Do test put rejects unknown model."""
    client, _ = client
    response = client.put(
        "/settings",
        json={"provider": "groq", "model": "gpt-4o", "api_key": "sk-test"},
    )
    assert response.status_code == 400
    assert "Unsupported model" in response.get_json()["error"]


def test_put_rejects_missing_key(client):
    """Do test put rejects missing key."""
    client, _ = client
    response = client.put("/settings", json={"provider": "groq", "model": "llama-3.3-70b-versatile"})
    assert response.status_code == 400
    assert "API key" in response.get_json()["error"]


def test_put_saves_encrypted_and_get_masks(client, demo_user, jwt_secret):
    """Do test put saves encrypted and get masks."""
    client, app_module = client
    plaintext = "sk-test-1234567890"
    response = client.put(
        "/settings",
        json={"provider": "groq", "model": "openai/gpt-oss-120b", "api_key": plaintext},
    )
    assert response.status_code == 200

    fetched = client.get("/settings").get_json()
    assert fetched["provider"] == "groq"
    assert fetched["model"] == "openai/gpt-oss-120b"
    assert fetched["masked_key"] == "••••7890"
    assert plaintext not in fetched["masked_key"]

    # The stored value must be ciphertext, decryptable back to the key,
    # attached to the demo user's record.
    stored = app_module.repository.get_user_settings(demo_user)
    assert stored["encrypted_api_key"] != plaintext
    assert decrypt_api_key(stored["encrypted_api_key"]) == plaintext


def test_put_requires_auth_resolved_user_not_found(app_module, repo, monkeypatch):
    """A valid token whose user no longer exists yields 401, not a crash."""
    monkeypatch.setattr(app_module, "repository", repo)
    monkeypatch.setenv("DEMO_MODE", "false")
    token = app_module.issue_token("ghost@papermind.local")
    with app_module.app.test_client() as client:
        response = client.get(
            "/settings", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 401


def test_verify_reports_ok(client, monkeypatch):
    """Do test verify reports ok."""
    client, app_module = client
    calls = []

    def fake_verify(provider, model, api_key):
        """Do fake verify."""
        calls.append((provider, model, api_key))
        return True, None

    monkeypatch.setattr(app_module, "verify_api_key", fake_verify)
    client.put(
        "/settings",
        json={"provider": "google", "model": "gemini-2.0-flash", "api_key": "sk-live"},
    )
    response = client.post("/settings/verify")
    assert response.status_code == 200
    assert response.get_json() == {"ok": True, "error": None}
    assert calls == [("google", "gemini-2.0-flash", "sk-live")]


def test_verify_reports_failure(client, monkeypatch):
    """Do test verify reports failure."""
    client, app_module = client
    monkeypatch.setattr(
        app_module,
        "verify_api_key",
        lambda p, m, k: (False, "401 invalid API key"),
    )
    client.put(
        "/settings",
        json={"provider": "groq", "model": "llama-3.3-70b-versatile", "api_key": "bad"},
    )
    response = client.post("/settings/verify")
    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is False
    assert "401" in body["error"]


def test_verify_without_settings(client):
    """Do test verify without settings."""
    client, _ = client
    response = client.post("/settings/verify")
    assert response.status_code == 400
    assert "No chat settings saved" in response.get_json()["error"]


def test_verify_error_never_contains_the_key(client, monkeypatch):
    """Do test verify error never contains the key."""
    client, app_module = client
    secret = "sk-super-secret-value"
    monkeypatch.setattr(
        app_module,
        "verify_api_key",
        lambda p, m, k: (False, f"401: key {k} rejected"),
    )
    client.put(
        "/settings",
        json={"provider": "groq", "model": "llama-3.3-70b-versatile", "api_key": secret},
    )
    body = client.post("/settings/verify").get_json()
    assert secret not in body["error"]


def test_token_user_gets_their_own_record(app_module, repo, monkeypatch, jwt_secret):
    """Outside demo mode, settings follow the token's user, not the demo user."""
    monkeypatch.setattr(app_module, "repository", repo)
    monkeypatch.setenv("DEMO_MODE", "false")
    with repo._session_factory() as session, session.begin():
        user = User(email="real@papermind.local", password_hash="x")
        session.add(user)
        session.flush()
        user_id = user.id
    token = app_module.issue_token("real@papermind.local")

    with app_module.app.test_client() as client:
        headers = {"Authorization": f"Bearer {token}"}
        response = client.put(
            "/settings",
            json={"provider": "groq", "model": "llama-3.3-70b-versatile", "api_key": "sk-u"},
            headers=headers,
        )
        assert response.status_code == 200
        assert repo.get_user_settings(user_id)["provider"] == "groq"
        assert client.get("/settings", headers=headers).get_json()["provider"] == "groq"