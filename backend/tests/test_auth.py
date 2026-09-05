"""
Auth tests: hashing, token issue/verify, demo bypass.

Auth flows run against a fake repository (dict-backed users) so no database
is touched; endpoint tests go through the Flask test client.
"""

import datetime

import jwt as pyjwt
import pytest

import app as app_module
from services import auth_service
from services.auth_service import (
    AuthError,
    hash_password,
    issue_token,
    verify_password,
    verify_token,
)


class FakeUserRepository:
    """FakeUserRepository."""

    def __init__(self):
        """Initialize."""
        self.users = {}

    def create_user(self, email, password_hash):
        """Do create user."""
        self.users[email] = password_hash
        return {"id": "u1", "email": email}

    def get_user_by_email(self, email):
        """Do get user by email."""
        if email not in self.users:
            return None
        return {"id": "u1", "email": email, "password_hash": self.users[email]}

    def list_files(self):
        """Do list files."""
        return []


@pytest.fixture
def fake_repo(monkeypatch):
    """Do fake repo."""
    repo = FakeUserRepository()
    monkeypatch.setattr(app_module, "repository", repo)
    return repo


@pytest.fixture
def client(fake_repo, monkeypatch):
    # Endpoints behind the auth tests must not touch real dependencies.
    """Do client."""

    class FakeStorage:
        """FakeStorage."""

        def list(self):
            """Do list."""
            return []

        def url(self, filename):
            """Do url."""
            return f"/storage/{filename}"

    monkeypatch.setattr(app_module, "storage", FakeStorage())
    monkeypatch.setattr(app_module.VectorService, "delete_all", lambda: None)
    with app_module.app.test_client() as client:
        yield client


def auth_headers(email="a@b.com"):
    """Do auth headers."""
    return {"Authorization": f"Bearer {issue_token(email)}"}


# -- hashing ----------------------------------------------------------------


def test_hash_is_not_plaintext_and_verifies():
    """Do test hash is not plaintext and verifies."""
    stored = hash_password("hunter2")
    assert stored != "hunter2"
    assert verify_password("hunter2", stored)
    assert not verify_password("wrong", stored)


def test_hashes_are_salted():
    """Do test hashes are salted."""
    assert hash_password("hunter2") != hash_password("hunter2")


# -- tokens -----------------------------------------------------------------


def test_token_roundtrip():
    """Do test token roundtrip."""
    token = issue_token("a@b.com")
    assert verify_token(token) == "a@b.com"


def test_garbage_token_rejected():
    """Do test garbage token rejected."""
    with pytest.raises(AuthError):
        verify_token("not-a-jwt")


def test_wrong_key_rejected(monkeypatch):
    """Do test wrong key rejected."""
    token = issue_token("a@b.com")
    monkeypatch.setenv("JWT_SECRET", "a-different-secret")
    with pytest.raises(AuthError):
        verify_token(token)


def test_expired_token_rejected(monkeypatch):
    """Do test expired token rejected."""
    monkeypatch.setenv("JWT_SECRET", "fixed-secret")
    now = datetime.datetime.now(datetime.timezone.utc)
    expired = pyjwt.encode(
        {"sub": "a@b.com", "exp": now - datetime.timedelta(minutes=1)},
        "fixed-secret",
        algorithm="HS256",
    )
    with pytest.raises(AuthError, match="expired"):
        verify_token(expired)


# -- register / login -------------------------------------------------------


def test_register_creates_user_with_bcrypt_hash_and_returns_token(client, fake_repo):
    """Do test register creates user with bcrypt hash and returns token."""
    response = client.post(
        "/auth/register", json={"email": "A@B.com", "password": "hunter2"}
    )

    assert response.status_code == 201
    assert response.get_json()["token"]
    stored = fake_repo.users["a@b.com"]  # normalized to lowercase
    assert stored != "hunter2"
    assert verify_password("hunter2", stored)


def test_register_rejects_duplicate_email(client, fake_repo):
    """Do test register rejects duplicate email."""
    client.post("/auth/register", json={"email": "a@b.com", "password": "x"})
    response = client.post("/auth/register", json={"email": "a@b.com", "password": "y"})
    assert response.status_code == 409


def test_register_requires_email_and_password(client):
    """Do test register requires email and password."""
    assert client.post("/auth/register", json={}).status_code == 400


def test_login_returns_usable_token(client, fake_repo):
    """Do test login returns usable token."""
    client.post("/auth/register", json={"email": "a@b.com", "password": "hunter2"})

    response = client.post(
        "/auth/login", json={"email": "a@b.com", "password": "hunter2"}
    )

    assert response.status_code == 200
    assert verify_token(response.get_json()["token"]) == "a@b.com"


def test_login_rejects_bad_credentials(client, fake_repo):
    """Do test login rejects bad credentials."""
    client.post("/auth/register", json={"email": "a@b.com", "password": "hunter2"})
    assert (
        client.post(
            "/auth/login", json={"email": "a@b.com", "password": "nope"}
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/auth/login", json={"email": "nobody@x.com", "password": "hunter2"}
        ).status_code
        == 401
    )


# -- endpoint enforcement ---------------------------------------------------


def test_demo_mode_allows_protected_endpoints_without_token(client, monkeypatch):
    """Do test demo mode allows protected endpoints without token."""
    monkeypatch.setenv("DEMO_MODE", "true")
    assert client.get("/files").status_code == 200


def test_demo_off_rejects_missing_token(client, monkeypatch):
    """Do test demo off rejects missing token."""
    monkeypatch.setenv("DEMO_MODE", "false")
    response = client.get("/files")
    assert response.status_code == 401
    assert "token" in response.get_json()["error"].lower()


def test_demo_off_rejects_invalid_token(client, monkeypatch):
    """Do test demo off rejects invalid token."""
    monkeypatch.setenv("DEMO_MODE", "false")
    response = client.get("/files", headers={"Authorization": "Bearer garbage"})
    assert response.status_code == 401


def test_demo_off_accepts_valid_token(client, monkeypatch):
    """Do test demo off accepts valid token."""
    monkeypatch.setenv("DEMO_MODE", "false")
    assert client.get("/files", headers=auth_headers()).status_code == 200


def test_wipe_requires_post_and_auth(client, monkeypatch):
    """Do test wipe requires post and auth."""
    monkeypatch.setenv("DEMO_MODE", "false")
    assert client.get("/delete-embeddings").status_code == 405
    assert client.post("/delete-embeddings").status_code == 401

    monkeypatch.setenv("DEMO_MODE", "true")
    assert client.post("/delete-embeddings").status_code == 200


def test_storage_download_is_gated(client, monkeypatch):
    """Do test storage download is gated."""
    monkeypatch.setenv("DEMO_MODE", "false")
    assert client.get("/storage/doc.pdf").status_code == 401

    monkeypatch.setenv("DEMO_MODE", "true")
    # 404 (not 401): demo on lets the request through to the storage layer.
    assert client.get("/storage/doc.pdf").status_code == 404
