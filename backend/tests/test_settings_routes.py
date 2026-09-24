"""
Settings API route tests — global single-row App Settings.

Run through the Flask test client with the same fakes as the flow tests:
in-memory sqlite repository, no real provider calls (the verifier is a
fake injected through the app factory). Every endpoint is open (no auth).
"""

import logging

import pytest
from dataclasses import replace
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import create_app
from composition import Services
from db import Base
from repositories import build_repositories
from services.accounts.secrets_service import (
    RESAVE_MESSAGE,
    decrypt_api_key,
    encrypt_api_key,
)
from services.llm.base import ChatCredentials


def _successful_verification(credentials):
    """Accept a candidate without making a provider call."""
    return True, None


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
    """App composed over in-memory repositories with a successful verifier."""
    application = make_app(repositories, settings_obj)
    with application.test_client() as client:
        yield client


def make_app(
    repositories,
    settings_obj,
    api_key_verifier=_successful_verification,
):
    """Compose a settings app, optionally with a fake key verifier."""
    services = replace(
        Services.from_settings(settings_obj),
        repositories=repositories,
        api_key_verifier=api_key_verifier,
    )
    return create_app(settings_obj, services=services)


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (
            RuntimeError("401 invalid API key candidate-secret"),
            "provider rejected the API key",
        ),
        (
            RuntimeError("model candidate-secret not found"),
            "provider could not use this model",
        ),
        (RuntimeError("candidate-secret timed out"), "provider did not respond"),
        (
            RuntimeError("model temporarily unavailable for candidate-secret"),
            "provider did not respond",
        ),
        (RuntimeError("candidate-secret"), "provider could not verify these settings"),
    ],
)
def test_verification_failures_use_key_free_categories(failure, expected):
    """Verification messages classify failures without repeating provider text."""
    from services.accounts.chat_settings_service import verification_error_message

    message = verification_error_message(failure)

    assert message == expected or expected in message
    assert "candidate-secret" not in message


def test_short_api_keys_are_also_scrubbed(monkeypatch):
    """Even a short secret is removed from a provider error."""
    from services.llm import factory
    from services.accounts.chat_settings_service import verify_api_key

    secret = "abc"
    monkeypatch.setattr(
        factory,
        "build_chat_provider",
        lambda credentials, **kwargs: type(
            "Provider",
            (),
            {"verify": lambda self: (_ for _ in ()).throw(RuntimeError(secret))},
        )(),
    )

    _, error = verify_api_key(
        ChatCredentials(provider="groq", model="openai/gpt-oss-20b", api_key=secret)
    )

    assert secret not in (error or "")


def test_verify_api_key_sanitizes_provider_errors(monkeypatch):
    """The verifier returns a safe error even when a provider echoes the key."""
    from services.llm import factory

    secret = "sk-verification-secret"

    class FailingProvider:
        def verify(self):
            raise RuntimeError(f"provider rejected {secret}")

    monkeypatch.setattr(
        factory,
        "build_chat_provider",
        lambda credentials, **kwargs: FailingProvider(),
    )

    from services.accounts.chat_settings_service import verify_api_key

    ok, error = verify_api_key(
        ChatCredentials(provider="groq", model="openai/gpt-oss-20b", api_key=secret)
    )

    assert ok is False
    assert secret not in (error or "")


def test_verification_exceptions_are_safe_in_responses_and_logs(
    repositories, app_secret, settings_obj, caplog
):
    """An unexpected verifier error cannot disclose the candidate key."""
    secret = "sk-exception-secret"

    def raise_secret(credentials):
        raise RuntimeError(f"provider included {secret}")

    application = make_app(
        repositories,
        settings_obj,
        api_key_verifier=raise_secret,
    )
    with caplog.at_level(logging.ERROR):
        with application.test_client() as client:
            response = client.post(
                "/settings/verify",
                json={
                    "provider": "groq",
                    "model": "openai/gpt-oss-20b",
                    "api_key": secret,
                },
            )

    assert response.status_code == 200
    assert response.get_json()["ok"] is False
    assert secret not in response.get_data(as_text=True)
    assert secret not in caplog.text


def test_chat_credentials_repr_redacts_the_api_key():
    """Logging a credentials object cannot reveal its secret."""
    credentials = ChatCredentials(
        provider="groq",
        model="openai/gpt-oss-120b",
        api_key="sk-never-log-this",
    )

    rendered = repr(credentials)

    assert "sk-never-log-this" not in rendered
    assert "api_key='••••'" in rendered


def test_recovery_message_distinguishes_missing_settings():
    """An empty workspace does not hear about saved settings."""
    from routes.settings import _recovery_message

    assert "No saved settings were affected" in _recovery_message("detail", False)
    assert "left unchanged" in _recovery_message("detail", True)


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
    """Empty settings return the model catalog and no provider."""
    response = client.get("/settings")
    assert response.status_code == 200
    body = response.get_json()
    assert body["provider"] is None
    assert body["model"] is None
    assert body["masked_key"] is None
    assert set(body["supported_models"]) == {"google", "groq"}


def test_get_settings_exposes_current_model_capabilities(client):
    """The catalog contains active models with complete capability data."""
    catalog = {
        (model["provider"], model["id"]): model
        for models in client.get("/settings").get_json()["supported_models"].values()
        for model in models
    }

    assert set(catalog) == {
        ("google", "gemini-2.5-flash"),
        ("google", "gemini-3.5-flash"),
        ("groq", "openai/gpt-oss-20b"),
        ("groq", "openai/gpt-oss-120b"),
    }
    assert catalog[("google", "gemini-2.5-flash")] == {
        "provider": "google",
        "id": "gemini-2.5-flash",
        "context_window_tokens": 1_048_576,
        "max_output_tokens": 65_536,
        "structured_output": True,
        "tool_use": True,
        "input_cost_per_million_usd": 0.30,
        "output_cost_per_million_usd": 2.50,
        "pricing_tier": "standard",
        "data_location": "cloud",
        "timeout_seconds": 15.0,
    }
    assert catalog[("google", "gemini-3.5-flash")]["input_cost_per_million_usd"] == 1.50
    assert (
        catalog[("google", "gemini-3.5-flash")]["output_cost_per_million_usd"] == 9.00
    )
    assert catalog[("groq", "openai/gpt-oss-120b")] == {
        "provider": "groq",
        "id": "openai/gpt-oss-120b",
        "context_window_tokens": 131_072,
        "max_output_tokens": 65_536,
        "structured_output": True,
        "tool_use": True,
        "input_cost_per_million_usd": 0.15,
        "output_cost_per_million_usd": 0.60,
        "pricing_tier": "standard",
        "data_location": "cloud",
        "timeout_seconds": 15.0,
    }


def test_get_settings_empty_state_no_auth_header(client):
    """GET /settings without Authorization header still returns 200."""
    response = client.get("/settings", headers={})
    assert response.status_code == 200


def test_save_rejects_a_malformed_candidate(client):
    """A first candidate failure explains that nothing was saved."""
    response = client.put(
        "/settings",
        json={"provider": "groq", "model": "openai/gpt-oss-20b"},
    )

    assert response.status_code == 400
    assert "not changed" in response.get_json()["error"]
    assert "supported_models" in response.get_json()


def test_malformed_candidate_is_rejected_before_storage_read(
    client, repositories, monkeypatch
):
    """Malformed settings return 400 even when storage is unavailable."""
    monkeypatch.setattr(
        repositories.app_settings,
        "get_app_settings",
        lambda: (_ for _ in ()).throw(RuntimeError("database unavailable")),
    )

    response = client.put("/settings", json=[])

    assert response.status_code == 400


def test_get_settings_flags_an_unsupported_saved_model(
    repositories, app_secret, settings_obj
):
    """A retired stored model is surfaced with recovery data."""
    repositories.app_settings.upsert_app_settings(
        "google",
        "gemini-2.0-flash",
        encrypt_api_key("sk-retired-model-key"),
    )
    application = make_app(repositories, settings_obj)
    with application.test_client() as client:
        response = client.get("/settings")

    assert response.status_code == 400
    body = response.get_json()
    assert body["provider"] == "google"
    assert body["model"] == "gemini-2.0-flash"
    assert body["needs_resave"] is True
    assert body["supported_models"]["google"]


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


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"provider": "google", "model": "gemini-2.5-flash"},
        {
            "provider": "google",
            "model": "gemini-2.5-flash",
            "api_key": 123,
        },
    ],
)
def test_settings_reject_malformed_candidates(client, payload):
    """Malformed settings requests fail before any provider or storage work."""
    assert client.put("/settings", json=payload).status_code == 400
    assert client.post("/settings/verify", json=payload).status_code == 400


def test_put_rejects_missing_key(client):
    """Do test put rejects missing key."""
    response = client.put(
        "/settings", json={"provider": "groq", "model": "openai/gpt-oss-20b"}
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


def test_failed_save_verification_keeps_working_settings(
    repositories, app_secret, settings_obj
):
    """A rejected candidate never replaces the saved provider or key."""
    repositories.app_settings.upsert_app_settings(
        "google",
        "gemini-2.5-flash",
        encrypt_api_key("sk-working-key"),
    )
    calls = []

    def reject_candidate(credentials):
        calls.append(credentials)
        return False, "401 invalid API key"

    application = make_app(
        repositories,
        settings_obj,
        api_key_verifier=reject_candidate,
    )
    with application.test_client() as client:
        response = client.put(
            "/settings",
            json={
                "provider": "groq",
                "model": "openai/gpt-oss-120b",
                "api_key": "sk-bad-key",
            },
        )

    assert response.status_code == 400
    assert "not changed" in response.get_json()["error"]
    assert "left unchanged" in response.get_json()["error"]
    assert calls == [
        ChatCredentials(
            provider="groq",
            model="openai/gpt-oss-120b",
            api_key="sk-bad-key",
        )
    ]
    stored = repositories.app_settings.get_app_settings()
    assert stored["provider"] == "google"
    assert stored["model"] == "gemini-2.5-flash"
    assert decrypt_api_key(stored["encrypted_api_key"]) == "sk-working-key"


def test_save_verifies_before_persisting(
    repositories, app_secret, settings_obj, monkeypatch
):
    """The repository is called only after candidate verification succeeds."""
    repositories.app_settings.upsert_app_settings(
        "google",
        "gemini-2.5-flash",
        encrypt_api_key("sk-working-key"),
    )
    events = []
    original_upsert = repositories.app_settings.upsert_app_settings

    def verify(candidate):
        events.append("verify")
        assert repositories.app_settings.get_app_settings()["model"] == (
            "gemini-2.5-flash"
        )
        return True, None

    def upsert(provider, model, encrypted):
        events.append("persist")
        return original_upsert(provider, model, encrypted)

    monkeypatch.setattr(repositories.app_settings, "upsert_app_settings", upsert)
    application = make_app(repositories, settings_obj, api_key_verifier=verify)
    with application.test_client() as client:
        response = client.put(
            "/settings",
            json={
                "provider": "groq",
                "model": "openai/gpt-oss-120b",
                "api_key": "sk-new-key",
            },
        )

    assert response.status_code == 200
    assert events == ["verify", "persist"]


def test_verify_candidate_does_not_replace_saved_settings(
    repositories, app_secret, settings_obj
):
    """Candidate verification leaves the working configuration untouched."""
    repositories.app_settings.upsert_app_settings(
        "google",
        "gemini-2.5-flash",
        encrypt_api_key("sk-working-key"),
    )
    calls = []

    def fake_verify(credentials):
        calls.append(credentials)
        return True, None

    application = make_app(repositories, settings_obj, api_key_verifier=fake_verify)
    with application.test_client() as client:
        response = client.post(
            "/settings/verify",
            json={
                "provider": "groq",
                "model": "openai/gpt-oss-120b",
                "api_key": "sk-candidate-key",
            },
        )

    assert response.status_code == 200
    assert response.get_json()["ok"] is True
    assert response.get_json()["error"] is None
    assert response.get_json()["supported_models"]["groq"]
    assert calls == [
        ChatCredentials(
            provider="groq",
            model="openai/gpt-oss-120b",
            api_key="sk-candidate-key",
        )
    ]
    stored = repositories.app_settings.get_app_settings()
    assert stored["provider"] == "google"
    assert stored["model"] == "gemini-2.5-flash"
    assert decrypt_api_key(stored["encrypted_api_key"]) == "sk-working-key"


def test_candidate_verification_leaves_provider_caches_untouched(
    repositories, app_secret, settings_obj, monkeypatch
):
    """Candidate verification neither uses nor evicts cached clients."""
    from services.llm import google_provider, groq_provider

    cleared = []
    monkeypatch.setattr(groq_provider, "clear_cache", lambda: cleared.append("groq"))
    monkeypatch.setattr(
        google_provider, "clear_cache", lambda: cleared.append("google")
    )
    application = make_app(repositories, settings_obj)
    with application.test_client() as client:
        response = client.post(
            "/settings/verify",
            json={
                "provider": "groq",
                "model": "openai/gpt-oss-20b",
                "api_key": "sk-candidate-key",
            },
        )

    assert response.status_code == 200
    assert cleared == []


def test_production_verification_disables_provider_client_caching(monkeypatch):
    """Production verification builds an isolated provider client."""
    from services.llm import factory
    from services.accounts.chat_settings_service import verify_api_key

    cache_flags = []

    class UncachedProvider:
        def verify(self):
            return None

    def fake_build(credentials, use_cache=True):
        cache_flags.append(use_cache)
        return UncachedProvider()

    monkeypatch.setattr(factory, "build_chat_provider", fake_build)

    assert verify_api_key(
        ChatCredentials(provider="groq", model="openai/gpt-oss-20b", api_key="key")
    ) == (True, None)
    assert cache_flags == [False]


def test_cache_cleanup_attempts_both_providers(
    repositories, app_secret, settings_obj, monkeypatch
):
    """One cache-clear failure does not skip the other provider."""
    from services.llm import google_provider, groq_provider

    cleared = []
    monkeypatch.setattr(
        groq_provider,
        "clear_cache",
        lambda: (_ for _ in ()).throw(RuntimeError("groq cache unavailable")),
    )
    monkeypatch.setattr(
        google_provider,
        "clear_cache",
        lambda: cleared.append("google"),
    )
    application = make_app(repositories, settings_obj)
    with application.test_client() as client:
        response = client.put(
            "/settings",
            json={
                "provider": "groq",
                "model": "openai/gpt-oss-20b",
                "api_key": "sk-candidate-key",
            },
        )

    assert response.status_code == 200
    assert cleared == ["google"]


def test_verify_reports_ok(repositories, app_secret, settings_obj):
    """Do test verify reports ok."""
    calls = []

    def fake_verify(credentials):
        """Do fake verify."""
        calls.append(credentials)
        return True, None

    application = make_app(repositories, settings_obj, api_key_verifier=fake_verify)
    with application.test_client() as client:
        response = client.post(
            "/settings/verify",
            json={
                "provider": "google",
                "model": "gemini-2.5-flash",
                "api_key": "sk-live",
            },
        )
    assert response.status_code == 200
    assert response.get_json()["ok"] is True
    assert response.get_json()["error"] is None
    assert response.get_json()["supported_models"]["google"]
    assert calls == [
        ChatCredentials(provider="google", model="gemini-2.5-flash", api_key="sk-live")
    ]


def test_verify_reports_failure(repositories, app_secret, settings_obj):
    """Do test verify reports failure."""
    application = make_app(
        repositories,
        settings_obj,
        api_key_verifier=lambda credentials: (False, "401 invalid API key"),
    )
    with application.test_client() as client:
        response = client.post(
            "/settings/verify",
            json={
                "provider": "groq",
                "model": "openai/gpt-oss-20b",
                "api_key": "bad",
            },
        )
    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is False
    assert "provider rejected the API key" in body["error"]
    assert "not changed" in body["error"]
    assert "No saved settings were affected" in body["error"]
    assert body["supported_models"]["groq"]


def test_verify_rejects_a_missing_candidate(client):
    """Candidate verification requires a complete settings object."""
    response = client.post("/settings/verify")
    assert response.status_code == 400
    assert response.get_json()["ok"] is False
    assert "JSON settings object" in response.get_json()["error"]
    assert response.get_json()["supported_models"]["google"]


def test_verify_error_never_contains_the_key(repositories, app_secret, settings_obj):
    """Do test verify error never contains the key."""
    secret = "sk-super-secret-value"

    def leaking_verify(credentials):
        """Echo the key back, as a misbehaving provider client might."""
        return False, f"401: key {credentials.api_key} rejected"

    application = make_app(repositories, settings_obj, api_key_verifier=leaking_verify)
    with application.test_client() as client:
        response = client.post(
            "/settings/verify",
            json={
                "provider": "groq",
                "model": "openai/gpt-oss-20b",
                "api_key": secret,
            },
        )
    assert secret not in response.get_data(as_text=True)


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
        "groq", "openai/gpt-oss-20b", _legacy_encrypt(plaintext, secret)
    )
    return plaintext


def test_get_settings_old_row_returns_resave_400(
    repositories, app_secret, settings_obj
):
    """GET /settings on an old row returns 400 with a re-save message."""
    application = make_app(repositories, settings_obj)
    plaintext = _seed_legacy_row(repositories, app_secret)
    with application.test_client() as client:
        response = client.get("/settings")
    assert response.status_code == 400
    body = response.get_json()
    assert body["error"] == RESAVE_MESSAGE
    assert body["needs_resave"] is True
    assert set(body["supported_models"]["google"][0]) == {
        "provider",
        "id",
        "context_window_tokens",
        "max_output_tokens",
        "structured_output",
        "tool_use",
        "input_cost_per_million_usd",
        "output_cost_per_million_usd",
        "pricing_tier",
        "data_location",
        "timeout_seconds",
    }
    assert plaintext not in body["error"]


def test_candidate_can_replace_an_old_encryption_row(
    repositories, app_secret, settings_obj
):
    """A verified candidate repairs a key encrypted with the old derivation."""
    application = make_app(repositories, settings_obj)
    _seed_legacy_row(repositories, app_secret)
    with application.test_client() as client:
        response = client.put(
            "/settings",
            json={
                "provider": "google",
                "model": "gemini-2.5-flash",
                "api_key": "sk-current-key",
            },
        )
        fetched = client.get("/settings")

    assert response.status_code == 200
    assert fetched.status_code == 200
    stored = repositories.app_settings.get_app_settings()
    assert stored["provider"] == "google"
    assert stored["model"] == "gemini-2.5-flash"
    assert decrypt_api_key(stored["encrypted_api_key"]) == "sk-current-key"


def test_get_repository_failure_still_returns_the_catalog(
    client, repositories, monkeypatch
):
    """A settings read failure leaves enough data to choose a replacement."""
    monkeypatch.setattr(
        repositories.app_settings,
        "get_app_settings",
        lambda: (_ for _ in ()).throw(RuntimeError("database unavailable")),
    )

    response = client.get("/settings")

    assert response.status_code == 500
    assert "Re-save" in response.get_json()["error"]
    assert response.get_json()["supported_models"]["google"]


def test_verify_repository_failure_returns_a_stable_error(
    client, repositories, monkeypatch
):
    """Verification depends on a readable settings state."""
    monkeypatch.setattr(
        repositories.app_settings,
        "get_app_settings",
        lambda: (_ for _ in ()).throw(RuntimeError("database unavailable")),
    )

    response = client.post(
        "/settings/verify",
        json={
            "provider": "groq",
            "model": "openai/gpt-oss-20b",
            "api_key": "sk-candidate-key",
        },
    )

    assert response.status_code == 500
    assert response.get_json()["ok"] is False
    assert response.get_json()["supported_models"]["groq"]


def test_encrypt_failure_leaves_settings_empty(client, repositories, monkeypatch):
    """A verified candidate is not persisted when encryption fails."""
    monkeypatch.setattr(
        "routes.settings.encrypt_api_key",
        lambda key: (_ for _ in ()).throw(RuntimeError("encryption unavailable")),
    )

    response = client.put(
        "/settings",
        json={
            "provider": "groq",
            "model": "openai/gpt-oss-20b",
            "api_key": "sk-candidate-key",
        },
    )

    assert response.status_code == 500
    assert "not changed" in response.get_json()["error"]
    assert repositories.app_settings.get_app_settings() is None


def test_upsert_failure_leaves_settings_empty(client, repositories, monkeypatch):
    """A verified candidate is not partially persisted on storage failure."""
    monkeypatch.setattr(
        repositories.app_settings,
        "upsert_app_settings",
        lambda provider, model, encrypted: (_ for _ in ()).throw(
            RuntimeError("database unavailable")
        ),
    )

    response = client.put(
        "/settings",
        json={
            "provider": "groq",
            "model": "openai/gpt-oss-20b",
            "api_key": "sk-candidate-key",
        },
    )

    assert response.status_code == 500
    assert "not changed" in response.get_json()["error"]
    assert repositories.app_settings.get_app_settings() is None


@pytest.mark.parametrize("failure", ["encrypt", "upsert"])
def test_verified_candidate_failure_preserves_working_settings(
    repositories, app_secret, settings_obj, monkeypatch, failure
):
    """A working key survives a verified candidate that cannot be stored."""
    repositories.app_settings.upsert_app_settings(
        "google",
        "gemini-2.5-flash",
        encrypt_api_key("sk-working-key"),
    )
    if failure == "encrypt":
        monkeypatch.setattr(
            "routes.settings.encrypt_api_key",
            lambda key: (_ for _ in ()).throw(RuntimeError("encryption unavailable")),
        )
    else:
        monkeypatch.setattr(
            repositories.app_settings,
            "upsert_app_settings",
            lambda provider, model, encrypted: (_ for _ in ()).throw(
                RuntimeError("database unavailable")
            ),
        )

    with make_app(repositories, settings_obj).test_client() as client:
        response = client.put(
            "/settings",
            json={
                "provider": "groq",
                "model": "openai/gpt-oss-20b",
                "api_key": "sk-candidate-key",
            },
        )

    assert response.status_code == 500
    stored = repositories.app_settings.get_app_settings()
    assert stored["provider"] == "google"
    assert stored["model"] == "gemini-2.5-flash"
    assert decrypt_api_key(stored["encrypted_api_key"]) == "sk-working-key"


def test_put_verifier_exception_stays_safe(repositories, app_secret, settings_obj):
    """An unexpected verifier failure cannot disclose the candidate."""
    secret = "sk-put-exception-secret"

    def raise_secret(credentials):
        raise RuntimeError(f"provider included {secret}")

    application = make_app(repositories, settings_obj, api_key_verifier=raise_secret)
    with application.test_client() as client:
        response = client.put(
            "/settings",
            json={
                "provider": "groq",
                "model": "openai/gpt-oss-20b",
                "api_key": secret,
            },
        )

    assert response.status_code == 400
    assert "provider could not verify" in response.get_json()["error"]
    assert secret not in response.get_data(as_text=True)


def test_response_old_row_returns_resave_400(repositories, app_secret, settings_obj):
    """POST /response on an old row returns 400 with a re-save message."""
    application = make_app(repositories, settings_obj)
    plaintext = _seed_legacy_row(repositories, app_secret)
    with application.test_client() as client:
        response = client.post("/response", json={"query": "hi", "filename": "doc.pdf"})
    assert response.status_code == 400
    body = response.get_json()
    assert body["error"] == RESAVE_MESSAGE
    assert body["needs_resave"] is True
    assert plaintext not in body["error"]


def test_put_is_idempotent_and_global(repositories, app_secret, settings_obj):
    """Second PUT overwrites the singleton row, not a per-user record."""
    application = make_app(repositories, settings_obj)
    with application.test_client() as client:
        first = client.put(
            "/settings",
            json={
                "provider": "groq",
                "model": "openai/gpt-oss-120b",
                "api_key": "sk-first",
            },
        )
        second = client.put(
            "/settings",
            json={
                "provider": "google",
                "model": "gemini-2.5-flash",
                "api_key": "sk-second",
            },
        )
        body = client.get("/settings").get_json()
    assert first.status_code == 200
    assert second.status_code == 200
    assert body["provider"] == "google"
    assert body["model"] == "gemini-2.5-flash"
    assert repositories.app_settings.get_app_settings()["provider"] == "google"
