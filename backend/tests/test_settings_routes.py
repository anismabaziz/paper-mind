"""
Settings API route tests.

Run through the Flask test client with the same fakes as the flow tests:
in-memory sqlite repository, no real provider calls (the verifier is a
fake injected through the app factory). Demo mode is on by default per
conftest; the auth test turns it off explicitly.
"""

import pytest
from dataclasses import replace
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import Services, create_app
from db import Base, Repository, User
from services.accounts.auth_service import issue_token
from services.accounts.secrets_service import decrypt_api_key
from services.llm.base import ChatCredentials


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
def jwt_secret(settings_obj, monkeypatch):
    """Set a deterministic JWT secret for crypto calls."""
    monkeypatch.setattr(settings_obj.auth, "jwt_secret", "test-secret")
    return "test-secret"


@pytest.fixture
def client(repo, demo_user, jwt_secret, settings_obj):
    """App composed over the in-memory repository with the default verifier."""
    services = replace(Services.from_settings(settings_obj), repository=repo)
    application = create_app(settings_obj, services=services)
    with application.test_client() as client:
        yield client


def make_app(repo, settings_obj, api_key_verifier=None):
    """Compose a settings app, optionally with a fake key verifier."""
    services = replace(
        Services.from_settings(settings_obj),
        repository=repo,
        api_key_verifier=api_key_verifier,
    )
    return create_app(settings_obj, services=services)


def test_settings_require_auth(repo, demo_user, settings_obj, monkeypatch):
    """Do test settings require auth."""
    monkeypatch.setattr(settings_obj.auth, "demo_mode", False)
    application = make_app(repo, settings_obj)
    with application.test_client() as client:
        for method, path in [
            (client.get, "/settings"),
            (client.put, "/settings"),
            (client.post, "/settings/verify"),
        ]:
            response = method(path)
            assert response.status_code == 401, path


def test_get_settings_empty_state(client):
    """Empty settings return the supported-models map and no provider."""
    response = client.get("/settings")
    assert response.status_code == 200
    body = response.get_json()
    assert body["provider"] is None
    assert body["model"] is None
    assert body["masked_key"] is None
    assert set(body["supported_models"]) == {"google", "groq"}


def test_put_rejects_unknown_provider(client):
    """Do test put rejects unknown provider."""
    response = client.put(
        "/settings",
        json={"provider": "openai", "model": "gpt-4o", "api_key": "sk-test"},
    )
    assert response.status_code == 400
    assert "Unsupported provider" in response.get_json()["error"]


def test_put_rejects_unknown_model(client):
    """Do test put rejects unknown model."""
    response = client.put(
        "/settings",
        json={"provider": "groq", "model": "gpt-4o", "api_key": "sk-test"},
    )
    assert response.status_code == 400
    assert "Unsupported model" in response.get_json()["error"]


def test_put_rejects_missing_key(client):
    """Do test put rejects missing key."""
    response = client.put(
        "/settings", json={"provider": "groq", "model": "llama-3.3-70b-versatile"}
    )
    assert response.status_code == 400
    assert "API key" in response.get_json()["error"]


def test_put_saves_encrypted_and_get_masks(client, repo, demo_user):
    """Do test put saves encrypted and get masks."""
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
    stored = repo.get_user_settings(demo_user)
    assert stored["encrypted_api_key"] != plaintext
    assert decrypt_api_key(stored["encrypted_api_key"]) == plaintext


def test_put_requires_auth_resolved_user_not_found(repo, settings_obj, monkeypatch):
    """A valid token whose user no longer exists yields 401, not a crash."""
    monkeypatch.setattr(settings_obj.auth, "demo_mode", False)
    token = issue_token("ghost@papermind.local")

    application = make_app(repo, settings_obj)
    with application.test_client() as client:
        response = client.get(
            "/settings", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 401


def test_verify_reports_ok(repo, demo_user, jwt_secret, settings_obj):
    """Do test verify reports ok."""
    calls = []

    def fake_verify(credentials):
        """Do fake verify."""
        calls.append(credentials)
        return True, None

    application = make_app(repo, settings_obj, api_key_verifier=fake_verify)
    with application.test_client() as client:
        client.put(
            "/settings",
            json={"provider": "google", "model": "gemini-2.0-flash", "api_key": "sk-live"},
        )
        response = client.post("/settings/verify")
    assert response.status_code == 200
    assert response.get_json() == {"ok": True, "error": None}
    assert calls == [
        ChatCredentials(provider="google", model="gemini-2.0-flash", api_key="sk-live")
    ]


def test_verify_reports_failure(repo, demo_user, jwt_secret, settings_obj):
    """Do test verify reports failure."""
    application = make_app(
        repo,
        settings_obj,
        api_key_verifier=lambda credentials: (False, "401 invalid API key"),
    )
    with application.test_client() as client:
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
    response = client.post("/settings/verify")
    assert response.status_code == 400
    assert "No chat settings saved" in response.get_json()["error"]


def test_verify_error_never_contains_the_key(
    repo, demo_user, jwt_secret, settings_obj
):
    """Do test verify error never contains the key."""
    secret = "sk-super-secret-value"

    def leaking_verify(credentials):
        """Echo the key back, as a misbehaving provider client might."""
        return False, f"401: key {credentials.api_key} rejected"

    application = make_app(repo, settings_obj, api_key_verifier=leaking_verify)
    with application.test_client() as client:
        client.put(
            "/settings",
            json={"provider": "groq", "model": "llama-3.3-70b-versatile", "api_key": secret},
        )
        body = client.post("/settings/verify").get_json()
    assert secret not in body["error"]


def test_token_user_gets_their_own_record(repo, jwt_secret, settings_obj, monkeypatch):
    """Outside demo mode, settings follow the token's user, not the demo user."""
    monkeypatch.setattr(settings_obj.auth, "demo_mode", False)
    with repo._session_factory() as session, session.begin():
        user = User(email="real@papermind.local", password_hash="x")
        session.add(user)
        session.flush()
        user_id = user.id
    token = issue_token("real@papermind.local")

    application = make_app(repo, settings_obj)
    with application.test_client() as client:
        headers = {"Authorization": f"Bearer {token}"}
        response = client.put(
            "/settings",
            json={"provider": "groq", "model": "llama-3.3-70b-versatile", "api_key": "sk-u"},
            headers=headers,
        )
        assert response.status_code == 200
        assert repo.get_user_settings(user_id)["provider"] == "groq"
        assert client.get("/settings", headers=headers).get_json()["provider"] == "groq"
