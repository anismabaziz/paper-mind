"""
Settings API route tests — global single-row App Settings.

Run through the Flask test client with the same fakes as the flow tests:
in-memory sqlite repository, no real provider calls (the verifier is a
fake injected through the app factory). Every endpoint is open (no auth).
"""

import pytest
from dataclasses import replace
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import create_app
from composition import Services
from db import Base
from repositories import build_repositories
from services.accounts.secrets_service import RESAVE_MESSAGE, decrypt_api_key
from services.llm.base import ChatCredentials


@pytest.fixture
def repositories():
    """Build all aggregate repositories over one in-memory database."""
    engine = create_engine(
        "sqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return build_repositories(sessionmaker(bind=engine))


@pytest.fixture
def app_secret(settings_obj, monkeypatch):
    """Set a deterministic APP_SECRET for crypto calls."""
    monkeypatch.setattr(settings_obj.auth, "app_secret", "test-secret")
    return "test-secret"


@pytest.fixture
def client(repositories, app_secret, settings_obj):
    """App composed over in-memory repositories with the default verifier."""
    services = replace(
        Services.from_settings(settings_obj), repositories=repositories
    )
    application = create_app(settings_obj, services=services)
    with application.test_client() as client:
        yield client


def make_app(repositories, settings_obj, api_key_verifier=None):
    """Compose a settings app, optionally with a fake key verifier."""
    services = replace(
        Services.from_settings(settings_obj),
        repositories=repositories,
        api_key_verifier=api_key_verifier,
    )
    return create_app(settings_obj, services=services)


def test_settings_are_open_without_auth(repositories, settings_obj):
    """Every settings endpoint returns non-401 without a token."""
    application = make_app(repositories, settings_obj)
    with application.test_client() as client:
        for method, path in [
            (client.get, "/settings"),
            (client.put, "/settings"),
            (client.post, "/settings/verify"),
        ]:
            response = method(path)
            assert response.status_code != 401, path


def test_get_settings_empty_state(client):
    """Empty settings return the supported-models map and no provider."""
    response = client.get("/settings")
    assert response.status_code == 200
    body = response.get_json()
    assert body["provider"] is None
    assert body["model"] is None
    assert body["masked_key"] is None
    assert set(body["supported_models"]) == {"google", "groq"}


def test_get_settings_empty_state_no_auth_header(client):
    """GET /settings without Authorization header still returns 200."""
    response = client.get("/settings", headers={})
    assert response.status_code == 200


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


def test_put_saves_encrypted_and_get_masks(client, repositories):
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
    # attached to the global app_settings row.
    stored = repositories.app_settings.get_app_settings()
    assert stored is not None
    assert stored["encrypted_api_key"] != plaintext
    assert decrypt_api_key(stored["encrypted_api_key"]) == plaintext


def test_verify_reports_ok(repositories, app_secret, settings_obj):
    """Do test verify reports ok."""
    calls = []

    def fake_verify(credentials):
        """Do fake verify."""
        calls.append(credentials)
        return True, None

    application = make_app(repositories, settings_obj, api_key_verifier=fake_verify)
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


def test_verify_reports_failure(repositories, app_secret, settings_obj):
    """Do test verify reports failure."""
    application = make_app(
        repositories,
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


def test_verify_error_never_contains_the_key(repositories, app_secret, settings_obj):
    """Do test verify error never contains the key."""
    secret = "sk-super-secret-value"

    def leaking_verify(credentials):
        """Echo the key back, as a misbehaving provider client might."""
        return False, f"401: key {credentials.api_key} rejected"

    application = make_app(repositories, settings_obj, api_key_verifier=leaking_verify)
    with application.test_client() as client:
        client.put(
            "/settings",
            json={"provider": "groq", "model": "llama-3.3-70b-versatile", "api_key": secret},
        )
        body = client.post("/settings/verify").get_json()
    assert secret not in body["error"]


def _legacy_encrypt(plaintext: str, secret: str) -> str:
    """Encrypt with the pre-HKDF SHA-256 derivation (old rows on disk)."""
    import base64
    import hashlib

    from cryptography.fernet import Fernet

    digest = hashlib.sha256(secret.encode()).digest()
    return Fernet(base64.urlsafe_b64encode(digest)).encrypt(plaintext.encode()).decode()


def _seed_legacy_row(repositories, secret):
    """Store an old-derivation row; returns the plaintext (never persisted)."""
    plaintext = "sk-legacy-row-1234567890"
    repositories.app_settings.upsert_app_settings(
        "groq", "llama-3.3-70b-versatile", _legacy_encrypt(plaintext, secret)
    )
    return plaintext


def test_get_settings_old_row_returns_resave_400(repositories, app_secret, settings_obj):
    """GET /settings on an old row returns 400 with a re-save message."""
    application = make_app(repositories, settings_obj)
    plaintext = _seed_legacy_row(repositories, app_secret)
    with application.test_client() as client:
        response = client.get("/settings")
    assert response.status_code == 400
    body = response.get_json()
    assert body["error"] == RESAVE_MESSAGE
    assert plaintext not in body["error"]


def test_verify_old_row_returns_resave_400(repositories, app_secret, settings_obj):
    """POST /settings/verify on an old row returns 400 with a re-save message."""
    application = make_app(repositories, settings_obj)
    plaintext = _seed_legacy_row(repositories, app_secret)
    with application.test_client() as client:
        response = client.post("/settings/verify")
    assert response.status_code == 400
    body = response.get_json()
    assert body["error"] == RESAVE_MESSAGE
    assert plaintext not in body["error"]


def test_response_old_row_returns_resave_400(repositories, app_secret, settings_obj):
    """POST /response on an old row returns 400 with a re-save message."""
    application = make_app(repositories, settings_obj)
    plaintext = _seed_legacy_row(repositories, app_secret)
    with application.test_client() as client:
        response = client.post("/response", json={"query": "hi", "filename": "doc.pdf"})
    assert response.status_code == 400
    body = response.get_json()
    assert body["error"] == RESAVE_MESSAGE
    assert plaintext not in body["error"]


def test_put_is_idempotent_and_global(repositories, app_secret, settings_obj):
    """Second PUT overwrites the singleton row, not a per-user record."""
    application = make_app(repositories, settings_obj)
    with application.test_client() as client:
        client.put(
            "/settings",
            json={"provider": "groq", "model": "openai/gpt-oss-120b", "api_key": "sk-first"},
        )
        client.put(
            "/settings",
            json={"provider": "google", "model": "gemini-2.0-flash", "api_key": "sk-second"},
        )
        body = client.get("/settings").get_json()
        assert body["provider"] == "google"
        assert body["model"] == "gemini-2.0-flash"
        assert repositories.app_settings.get_app_settings()["provider"] == "google"
