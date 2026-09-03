"""Service-layer flow tests.

Every external edge is a fake: vector index, embeddings, LLM, and storage
are monkeypatched; the repository runs against an in-memory sqlite. No test
touches Pinecone, an LLM provider, a real Postgres, or the real upload dir.
"""

import importlib
import io
import os
import pathlib
import subprocess
import sys

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db import Base, Repository


@pytest.fixture
def app_module(monkeypatch):
    import app

    return importlib.import_module("app")


@pytest.fixture
def repo():
    engine = create_engine(
        "sqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return Repository(sessionmaker(bind=engine))


class FakeStorage:
    """In-memory stand-in for the storage seam."""

    def __init__(self):
        self.blobs = {}

    def save(self, filename, content):
        self.blobs[filename] = content

    def open(self, filename):
        return self.blobs[filename]

    def exists(self, filename):
        return filename in self.blobs

    def delete(self, filename):
        self.blobs.pop(filename, None)

    def list(self):
        return [{"name": name, "size": len(data)} for name, data in sorted(self.blobs.items())]

    def url(self, filename):
        return f"/storage/{filename}"


@pytest.fixture
def fake_storage(monkeypatch, app_module):
    storage = FakeStorage()
    monkeypatch.setattr(app_module, "storage", storage)
    return storage


class FakeVectorService:
    def __init__(self):
        self.upserts = []
        self.deleted = []
        self.deleted_all = False
        self.matches = ["chunk about topic"]

    def upsert_vectors(self, embeddings, texts, filename):
        self.upserts.append((embeddings, texts, filename))

    def query_vectors(self, embedding, filename, top_k=3):
        return self.matches

    def delete_by_filename(self, filename):
        self.deleted.append(filename)

    def delete_all(self):
        self.deleted_all = True


@pytest.fixture
def fake_vectors(monkeypatch, app_module):
    fake = FakeVectorService()
    monkeypatch.setattr(app_module.VectorService, "upsert_vectors", fake.upsert_vectors)
    monkeypatch.setattr(app_module.VectorService, "query_vectors", fake.query_vectors)
    monkeypatch.setattr(app_module.VectorService, "delete_by_filename", fake.delete_by_filename)
    monkeypatch.setattr(app_module.VectorService, "delete_all", fake.delete_all)
    return fake


@pytest.fixture
def fake_ai(monkeypatch, app_module):
    calls = {"embedded": [], "answered": []}

    def extract_text(pdf_content):
        return "fake document text"

    def split_text(text, chunk_size=600, chunk_overlap=100):
        return ["chunk about topic", "another chunk"]

    monkeypatch.setattr(app_module.PDFService, "extract_text", staticmethod(extract_text))
    monkeypatch.setattr(app_module.PDFService, "split_text", staticmethod(split_text))

    def get_embeddings(texts):
        if isinstance(texts, str):
            texts = [texts]
        calls["embedded"].extend(texts)
        return [[0.1, 0.2] for _ in texts]

    def generate_response(query, context):
        calls["answered"].append((query, context))
        return "The answer is 42."

    monkeypatch.setattr(app_module.AIService, "get_embeddings", staticmethod(get_embeddings))
    monkeypatch.setattr(app_module.AIService, "generate_response", staticmethod(generate_response))
    return calls


@pytest.fixture
def client(app_module, repo, fake_storage, fake_vectors, fake_ai):
    monkeypatch_fixture = pytest.MonkeyPatch()
    monkeypatch_fixture.setattr(app_module, "repository", repo)
    with app_module.app.test_client() as client:
        yield client
    monkeypatch_fixture.undo()


def upload(client, name="doc.pdf"):
    data = {"file": (io.BytesIO(b"%PDF-fake-bytes"), name)}
    return client.post("/upload", data=data, content_type="multipart/form-data")


def test_upload_stores_bytes_and_creates_record(client, fake_storage, repo):
    response = upload(client)

    assert response.status_code == 200
    body = response.get_json()
    filename = body["file"]["name"]
    assert fake_storage.blobs[filename] == b"%PDF-fake-bytes"
    assert body["file"]["url"] == f"http://localhost/storage/{filename}"

    listing = client.get("/files").get_json()["files"]
    assert [f["name"] for f in listing] == [filename]
    assert listing[0]["metadata"]["size"] == len(b"%PDF-fake-bytes")
    assert listing[0]["is_processed"] is False


def test_upload_without_file_is_rejected(client):
    assert client.post("/upload", data={}).status_code == 400


def test_process_embeds_and_marks_processed(client, fake_vectors, fake_ai):
    filename = upload(client).get_json()["file"]["name"]

    response = client.post("/process-file", json={"filename": filename})

    assert response.status_code == 200
    assert fake_vectors.upserts, "document should reach the vector index"
    _, texts, upserted_file = fake_vectors.upserts[0]
    assert upserted_file == filename
    assert texts, "chunks should be extracted before embedding"
    assert fake_ai["embedded"], "chunks should be embedded"
    assert client.post("/file/is-processed", json={"filename": filename}).get_json()[
        "is_processed"
    ] is True


def test_ask_stores_messages_and_returns_answer(client, fake_vectors, fake_ai):
    filename = upload(client).get_json()["file"]["name"]
    client.post("/process-file", json={"filename": filename})

    response = client.post("/response", json={"query": "what?", "filename": filename})

    assert response.status_code == 200
    assert response.get_json()["results"] == "The answer is 42."
    assert fake_ai["answered"][0][0] == "what?"

    history = client.get(f"/messages?filename={filename}").get_json()["messages"]
    assert [(m["sender"], m["text"]) for m in history] == [
        ("user", "what?"),
        ("bot", "The answer is 42."),
    ]


def test_delete_removes_everything_with_no_orphans(client, fake_storage, fake_vectors, repo):
    filename = upload(client).get_json()["file"]["name"]
    client.post("/process-file", json={"filename": filename})
    client.post("/response", json={"query": "what?", "filename": filename})

    response = client.delete(f"/files/remove?path={filename}")

    assert response.status_code == 200
    assert fake_storage.blobs == {}
    assert fake_vectors.deleted == [filename]
    assert repo.list_files() == []
    assert client.get(f"/messages?filename={filename}").get_json()["messages"] == []
    assert client.get("/files").get_json()["files"] == []


def test_fresh_database_reaches_current_schema_via_migrations(tmp_path):
    """Running the migrations on an empty database produces the app schema."""
    db_path = tmp_path / "fresh.db"
    backend_dir = pathlib.Path(__file__).resolve().parent.parent
    env = {
        # Restrict PATH/HOME so the run can't pick up a local .env via cwd
        **{k: v for k, v in os.environ.items() if k != "DATABASE_URL"},
        "DATABASE_URL": f"sqlite:///{db_path}",
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(pathlib.Path.home()),
    }
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        capture_output=True,
        text=True,
        cwd=backend_dir,
        env=env,
    )
    assert result.returncode == 0, result.stderr

    import sqlite3

    tables = {
        row[0]
        for row in sqlite3.connect(db_path).execute(
            "select name from sqlite_master where type='table'"
        )
    }
    assert {"files", "conversations", "messages"} <= tables
