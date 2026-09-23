"""
Secrets service and app settings repository.

Crypto tests exercise the Fernet round-trip and the APP_SECRET failure
modes; repository tests run against an in-memory SQLite database created
from the ORM metadata.
"""

import pytest

from services.accounts.secrets_service import (
    RESAVE_MESSAGE,
    SecretsError,
    SecretsResaveRequiredError,
    decrypt_api_key,
    encrypt_api_key,
)


@pytest.fixture
def app_secret(settings_obj, monkeypatch):
    """Set a deterministic APP_SECRET for crypto calls."""
    monkeypatch.setattr(settings_obj.auth, "app_secret", "test-secret")
    return "test-secret"


def test_round_trip(app_secret):
    """Do test round trip."""
    plaintext = "sk-live-abc123"
    ciphertext = encrypt_api_key(plaintext)
    assert ciphertext != plaintext
    assert decrypt_api_key(ciphertext) == plaintext


def test_missing_app_secret_fails_at_use(settings_obj, monkeypatch):
    """Do test missing app secret fails at use."""
    monkeypatch.setattr(settings_obj.auth, "app_secret", None)
    with pytest.raises(SecretsError, match="APP_SECRET"):
        encrypt_api_key("sk-test")


def test_wrong_secret_cannot_decrypt(app_secret, settings_obj, monkeypatch):
    """Do test wrong secret cannot decrypt."""
    ciphertext = encrypt_api_key("sk-test")
    monkeypatch.setattr(settings_obj.auth, "app_secret", "a-different-secret")
    with pytest.raises(SecretsError, match="decrypted"):
        decrypt_api_key(ciphertext)


def test_each_encryption_is_unique(app_secret):
    """Fernet nonces differ per call, but both decrypt to the plaintext."""
    first = encrypt_api_key("sk-test")
    second = encrypt_api_key("sk-test")
    assert first != second
    assert decrypt_api_key(first) == decrypt_api_key(second) == "sk-test"


def _legacy_encrypt(plaintext: str, secret: str) -> str:
    """Encrypt with the pre-HKDF SHA-256 derivation (old rows on disk)."""
    import base64
    import hashlib

    from cryptography.fernet import Fernet

    digest = hashlib.sha256(secret.encode()).digest()
    return Fernet(base64.urlsafe_b64encode(digest)).encrypt(plaintext.encode()).decode()


def test_old_sha256_row_raises_resave(app_secret):
    """Old derivation rows raise a re-save error instead of returning plaintext."""
    legacy = _legacy_encrypt("sk-live-abc123", app_secret)
    with pytest.raises(SecretsResaveRequiredError) as excinfo:
        decrypt_api_key(legacy)
    assert isinstance(excinfo.value, SecretsError)
    assert str(excinfo.value) == RESAVE_MESSAGE


def test_new_ciphertext_uses_hkdf_not_legacy_sha256(app_secret):
    """New ciphertext must not open with the legacy SHA-256 key."""
    import base64
    import hashlib

    from cryptography.fernet import Fernet, InvalidToken

    ciphertext = encrypt_api_key("sk-live-abc123")
    legacy_key = base64.urlsafe_b64encode(hashlib.sha256(app_secret.encode()).digest())
    with pytest.raises(InvalidToken):
        Fernet(legacy_key).decrypt(ciphertext.encode())
    assert decrypt_api_key(ciphertext) == "sk-live-abc123"


def test_app_settings_upsert_round_trip(app_secret):
    """Global app settings round-trip via repository."""
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool
    from sqlalchemy.orm import sessionmaker
    from db import Base, Repository

    engine = create_engine(
        "sqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    repo = Repository(sessionmaker(bind=engine))

    assert repo.get_app_settings() is None

    encrypted = encrypt_api_key("sk-live-abc123")
    repo.upsert_app_settings("groq", "llama-3", encrypted)
    stored = repo.get_app_settings()
    assert stored["provider"] == "groq"
    assert stored["model"] == "llama-3"
    assert stored["encrypted_api_key"] == encrypted
    assert decrypt_api_key(stored["encrypted_api_key"]) == "sk-live-abc123"

    repo.upsert_app_settings("google", "gemini-2.0-flash", encrypted)
    updated = repo.get_app_settings()
    assert updated["provider"] == "google"
    assert updated["model"] == "gemini-2.0-flash"
