"""
Secrets service and user settings repository.

Crypto tests exercise the Fernet round-trip and the JWT_SECRET failure
modes; repository tests run against an in-memory SQLite database created
from the ORM metadata.
"""

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db import Base, Repository, User
from services.secrets_service import (
    SecretsError,
    decrypt_api_key,
    encrypt_api_key,
)


@pytest.fixture
def repo_with_user():
    """Provide a repository with one user, and that user's id."""
    engine = create_engine(
        "sqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _enable_sqlite_fk(dbapi_connection, _):
        # ondelete CASCADE is enforced only with the pragma on.
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    repository = Repository(sessionmaker(bind=engine))
    with repository._session_factory() as session, session.begin():
        user = User(email="u@example.com", password_hash="x")
        session.add(user)
        session.flush()
        return repository, user.id


@pytest.fixture
def jwt_secret(settings_obj, monkeypatch):
    """Set a deterministic JWT secret for crypto calls."""
    monkeypatch.setattr(settings_obj.auth, "jwt_secret", "test-secret")
    return "test-secret"


def test_round_trip(jwt_secret):
    """Do test round trip."""
    plaintext = "sk-live-abc123"
    ciphertext = encrypt_api_key(plaintext)
    assert ciphertext != plaintext
    assert decrypt_api_key(ciphertext) == plaintext


def test_missing_jwt_secret_fails_at_use(settings_obj, monkeypatch):
    """Do test missing jwt secret fails at use."""
    monkeypatch.setattr(settings_obj.auth, "jwt_secret", None)
    with pytest.raises(SecretsError, match="JWT_SECRET"):
        encrypt_api_key("sk-test")


def test_wrong_secret_cannot_decrypt(jwt_secret, settings_obj, monkeypatch):
    """Do test wrong secret cannot decrypt."""
    ciphertext = encrypt_api_key("sk-test")
    monkeypatch.setattr(settings_obj.auth, "jwt_secret", "a-different-secret")
    with pytest.raises(SecretsError, match="decrypted"):
        decrypt_api_key(ciphertext)


def test_each_encryption_is_unique(jwt_secret):
    """Fernet nonces differ per call, but both decrypt to the plaintext."""
    first = encrypt_api_key("sk-test")
    second = encrypt_api_key("sk-test")
    assert first != second
    assert decrypt_api_key(first) == decrypt_api_key(second) == "sk-test"


def test_user_settings_upsert_round_trip(repo_with_user, jwt_secret):
    """Stored keys stay encrypted and survive an upsert that replaces fields."""
    repository, user = repo_with_user

    assert repository.get_user_settings(user) is None

    encrypted = encrypt_api_key("sk-live-abc123")
    assert encrypted != "sk-live-abc123"

    repository.upsert_user_settings(user, "groq", "llama-3", encrypted)
    stored = repository.get_user_settings(user)
    assert stored["user_id"] == user
    assert stored["provider"] == "groq"
    assert stored["model"] == "llama-3"
    assert stored["encrypted_api_key"] == encrypted

    repository.upsert_user_settings(user, "google", "gemini-2.0-flash", encrypted)
    updated = repository.get_user_settings(user)
    assert updated["provider"] == "google"
    assert updated["model"] == "gemini-2.0-flash"
    assert decrypt_api_key(updated["encrypted_api_key"]) == "sk-live-abc123"


def test_user_settings_cascade_delete_with_user(repo_with_user, jwt_secret):
    """Do test settings cascade delete with user."""
    repository, user = repo_with_user
    repository.upsert_user_settings(user, "groq", "llama-3", encrypt_api_key("sk"))
    with repository._session_factory() as session, session.begin():
        session.delete(session.get(User, user))
    assert repository.get_user_settings(user) is None
