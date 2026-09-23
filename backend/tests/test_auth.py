"""
Auth removal tests: every endpoint is open, no JWT is issued or verified.

Replaces the legacy multi-user auth tests. The app runs as a single-instance
open-source workspace; no token is required and no /auth/* routes exist.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from app import create_app
from composition import Services


class FakeStorage:
    """FakeStorage."""

    def list(self):
        """List."""
        return []

    def url(self, filename):
        """Url."""
        return f"/storage/{filename}"


class FakeVectorService:
    """Counts wipe requests; no real vectors."""

    def __init__(self):
        """Initialize."""
        self.deleted_all = False

    def delete_all(self):
        """Delete all."""
        self.deleted_all = True


@pytest.fixture
def fake_repositories():
    """Provide minimal aggregate repositories for open-endpoint checks."""
    class _Repo:
        def list_files(self):
            return []

        def get_app_settings(self):
            return None

        def get_file(self, filename):
            return None

    repository = _Repo()
    return SimpleNamespace(
        files=repository,
        app_settings=repository,
        conversations=repository,
    )


@pytest.fixture
def client(fake_repositories, settings_obj):
    """Provide a test client wired to fakes."""
    services = replace(
        Services.from_settings(settings_obj),
        repositories=fake_repositories,
        storage=FakeStorage(),
        vector_service=FakeVectorService(),
    )
    application = create_app(settings_obj, services=services)
    with application.test_client() as client:
        yield client


def test_open_endpoints_return_non_401_without_token(client):
    """All remaining routes return non-401 without any Authorization header."""
    # GET /files, GET /settings are 200 in open mode
    assert client.get("/files").status_code != 401
    assert client.get("/files").status_code == 200
    assert client.get("/settings").status_code != 401
    assert client.get("/settings").status_code == 200

    # POST /upload without file is 400, not 401
    assert client.post("/upload", data={}).status_code != 401

    # POST /response without provider is 400 (no settings), not 401
    resp = client.post("/response", json={"query": "hi", "filename": "doc.pdf"})
    assert resp.status_code != 401

    # POST /delete-embeddings is open
    assert client.post("/delete-embeddings").status_code != 401


def test_auth_routes_are_gone(client):
    """Legacy /auth/* routes no longer exist."""
    assert client.post("/auth/register", json={}).status_code == 404
    assert client.post("/auth/login", json={}).status_code == 404


def test_legacy_auth_module_is_removed():
    """The retired authentication module is no longer importable."""
    import importlib.util

    assert importlib.util.find_spec("services.accounts.auth_service") is None


def test_no_authorization_header_required_for_storage(client):
    """Storage download goes to 404 (not 401) when file missing."""
    # Without auth, missing file is 404, not 401
    assert client.get("/storage/doc.pdf").status_code == 404


def test_delete_embeddings_method_still_post(client):
    """DELETE embeddings still requires POST, but no auth."""
    assert client.get("/delete-embeddings").status_code == 405
    assert client.post("/delete-embeddings").status_code == 200
