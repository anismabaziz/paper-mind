"""Service-layer flow tests.

Every external edge is a fake: vector index, embeddings, LLM, and storage
are monkeypatched; the repository runs against an in-memory sqlite. No test
touches Pinecone, an LLM provider, a real Postgres, or the real upload dir.
"""

import config
import importlib
import io
import json
import os
import pathlib
import subprocess
import sys

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db import Base, Repository
from services.document_parser import DocumentParser
from services.pdf_service import PDFParser
from services.vector_service import shape_sources


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
        self.matches = [
            {
                "content": "chunk about topic",
                "document": "doc.pdf",
                "chunk_index": 0,
                "score": 0.92,
            }
        ]

    def upsert_vectors(self, embeddings, texts, filename):
        self.upserts.append((embeddings, texts, filename))

    def query_vectors(self, embedding, filename, top_k=3):
        # Route through the real shaping so flow tests see the same
        # dedupe/bound/order behavior as production retrieval.
        return shape_sources(self.matches)

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
    calls = {"embedded": [], "answered": [], "streamed": []}

    def extract_text(pdf_content):
        return "fake document text"

    def split_text(text, chunk_size=600, chunk_overlap=100):
        return ["chunk about topic", "another chunk"]

    monkeypatch.setattr(PDFParser, "extract_text", staticmethod(extract_text))
    monkeypatch.setattr(DocumentParser, "split_text", staticmethod(split_text))

    def get_embeddings(texts):
        if isinstance(texts, str):
            texts = [texts]
        calls["embedded"].extend(texts)
        return [[0.1, 0.2] for _ in texts]

    def generate_response(query, context):
        calls["answered"].append((query, context))
        return "The answer is 42."

    def stream_response(query, context):
        calls["streamed"].append((query, context))
        yield "The answer "
        yield "is 42."

    monkeypatch.setattr(app_module.AIService, "get_embeddings", staticmethod(get_embeddings))
    monkeypatch.setattr(app_module.AIService, "generate_response", staticmethod(generate_response))
    monkeypatch.setattr(app_module.AIService, "stream_response", staticmethod(stream_response))
    return calls


def parse_sse(body):
    """Split an SSE body into (event, data) tuples."""
    events = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        name, data = None, None
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        events.append((name, data))
    return events


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


def test_ask_streams_tokens_and_persists_sources(client, fake_vectors, fake_ai):
    filename = upload(client).get_json()["file"]["name"]
    client.post("/process-file", json={"filename": filename})

    response = client.post("/response", json={"query": "what?", "filename": filename})

    assert response.status_code == 200
    assert response.mimetype == "text/event-stream"
    events = parse_sse(response.get_data(as_text=True))

    tokens = [data["text"] for name, data in events if name == "token"]
    assert "".join(tokens) == "The answer is 42.", "tokens should arrive as fragments"

    done_name, done_data = events[-1]
    assert done_name == "done"
    assert done_data["done"] is True
    assert done_data["sources"] == [
        {
            "content": "chunk about topic",
            "document": "doc.pdf",
            "chunk_index": 0,
            "score": 0.92,
        }
    ]

    history = client.get(f"/messages?filename={filename}").get_json()["messages"]
    assert [(m["sender"], m["text"]) for m in history] == [
        ("user", "what?"),
        ("bot", "The answer is 42."),
    ]
    assert history[1]["sources"] == done_data["sources"], (
        "the terminal event must carry the sources persisted with the answer"
    )


def test_provider_failure_still_leaves_a_visible_reply(client, fake_vectors, fake_ai, app_module, capsys):
    filename = upload(client).get_json()["file"]["name"]
    client.post("/process-file", json={"filename": filename})

    def broken_stream(query, context):
        raise RuntimeError("provider down")
        yield  # pragma: no cover

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(app_module.AIService, "stream_response", staticmethod(broken_stream))
    try:
        response = client.post("/response", json={"query": "what?", "filename": filename})
    finally:
        monkeypatch.undo()

    events = parse_sse(response.get_data(as_text=True))
    names = [name for name, _ in events]
    assert "error" in names, "a provider failure must surface as an error event"

    error_data = next(data for name, data in events if name == "error")
    assert error_data["error"], "the error event should carry a readable message"

    history = client.get(f"/messages?filename={filename}").get_json()["messages"]
    assert [m["sender"] for m in history] == ["user", "bot"], (
        "a failed turn must still end with an assistant reply, not a stranded question"
    )
    assert error_data["error"] in history[-1]["text"]


def test_primary_provider_failure_falls_back_to_secondary(app_module, monkeypatch):
    """The primary provider failing before any token hands off to the fallback."""
    from services.ai_service import AIService
    from services.groq_service import GroqService
    from services.google_service import GoogleService

    monkeypatch.setattr(config, "MODE", "groq")

    def primary_stream(query, context):
        raise RuntimeError("primary down")
        yield  # pragma: no cover

    def fallback_stream(query, context):
        yield "fallback "

    monkeypatch.setattr(
        GroqService, "stream_response", staticmethod(primary_stream)
    )
    monkeypatch.setattr(
        GoogleService, "stream_response", staticmethod(fallback_stream)
    )

    tokens = list(AIService.stream_response("q", "ctx"))
    assert "".join(tokens) == "fallback "


def test_midstream_primary_failure_does_not_replay_via_fallback(app_module, monkeypatch):
    """A primary that dies mid-stream surfaces the error instead of restarting."""
    from services.ai_service import AIService
    from services.groq_service import GroqService
    from services.google_service import GoogleService

    monkeypatch.setattr(config, "MODE", "groq")

    def primary_stream(query, context):
        yield "partial"
        raise RuntimeError("midstream failure")

    fallback_calls = []

    def fallback_stream(query, context):
        fallback_calls.append((query, context))
        yield "fallback"

    monkeypatch.setattr(
        GroqService, "stream_response", staticmethod(primary_stream)
    )
    monkeypatch.setattr(
        GoogleService, "stream_response", staticmethod(fallback_stream)
    )

    with pytest.raises(RuntimeError):
        list(AIService.stream_response("q", "ctx"))

    assert fallback_calls == [], "fallback must not duplicate a half-streamed answer"


def test_primary_success_does_not_invoke_fallback(app_module, monkeypatch):
    """When the primary streams fully, the fallback must never be called."""
    from services.ai_service import AIService
    from services.groq_service import GroqService
    from services.google_service import GoogleService

    monkeypatch.setattr(config, "MODE", "groq")

    def primary_stream(query, context):
        yield "primary answer"

    fallback_calls = []

    def fallback_stream(query, context):
        fallback_calls.append((query, context))
        yield "fallback"

    monkeypatch.setattr(
        GroqService, "stream_response", staticmethod(primary_stream)
    )
    monkeypatch.setattr(
        GoogleService, "stream_response", staticmethod(fallback_stream)
    )

    tokens = list(AIService.stream_response("q", "ctx"))
    assert "".join(tokens) == "primary answer"
    assert fallback_calls == []


def test_sources_panel_order_matches_llm_context_order(client, fake_vectors, fake_ai):
    filename = upload(client).get_json()["file"]["name"]
    client.post("/process-file", json={"filename": filename})

    fake_vectors.matches = [
        {"content": "weak chunk", "document": "doc.pdf", "chunk_index": 2, "score": 0.31},
        {"content": "strong chunk", "document": "doc.pdf", "chunk_index": 1, "score": 0.95},
    ]

    response = client.post("/response", json={"query": "what?", "filename": filename})

    done_sources = parse_sse(response.get_data(as_text=True))[-1][1]["sources"]
    assert [s["content"] for s in done_sources] == ["strong chunk", "weak chunk"], (
        "sources must be ordered by score, not by index order"
    )

    # The LLM received the same chunks, in the same order, as its context.
    _, context = fake_ai["streamed"][0]
    assert context == "strong chunk\n\nweak chunk"

    history = client.get(f"/messages?filename={filename}").get_json()["messages"]
    assert history[1]["sources"] == done_sources, (
        "the persisted sources must replay in the same order for history"
    )


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
