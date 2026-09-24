"""
Route and storage hardening coverage.

External behavior only: HTTP status and payload for the file-meta and
download routes, the storage traversal guard, ingestion retry after an
embedding failure, the SSE error path, title backfill served through the
listing, and the health endpoint. This module uses fakes and in-memory
SQLite throughout.
"""

# ruff: noqa: D100, D101, D102, D103, D104, D105, D107

import io
from dataclasses import replace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import create_app
from composition import Services
from db import Base, FileRecord
from repositories import build_repositories
from tests.ingestion_helpers import build_test_worker
from services.accounts.secrets_service import encrypt_api_key
from services.parsing.document_parser import Chunk
from services.retrieval.base import RetrievalResult
from services.retrieval.hybrid import build_sparse_vector
from storage import LocalStorage
from tests.sse import parse_sse


@pytest.fixture
def repositories():
    engine = create_engine(
        "sqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    repositories = build_repositories(sessionmaker(bind=engine))
    repositories.app_settings.upsert_app_settings(
        "groq", "openai/gpt-oss-120b", encrypt_api_key("sk-test-chat-key")
    )
    return repositories


class FakeStorage:
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
        return [
            {"name": name, "size": len(data)}
            for name, data in sorted(self.blobs.items())
        ]

    def url(self, filename):
        return f"/storage/{filename}"


@pytest.fixture
def fake_storage():
    return FakeStorage()


class FakeVectorService:
    def __init__(self):
        self.upserts = []
        self.deleted = []

    def upsert_chunks(self, embeddings, chunks, filename, **kwargs):
        self.upserts.append((embeddings, chunks, filename))

    def query_vectors(self, embedding, filename, **kwargs):
        query_text = kwargs.get("query_text")
        method = (
            "hybrid"
            if query_text and build_sparse_vector(query_text)["indices"]
            else "dense"
        )
        return RetrievalResult(
            sources=[
                {
                    "content": "chunk about topic",
                    "document": filename,
                    "chunk_index": 0,
                    "score": 0.9,
                }
            ],
            method=method,
            outcome="success",
        )

    def delete_by_filename(self, filename):
        self.deleted.append(filename)

    def delete_all(self):
        pass


@pytest.fixture
def fake_vectors():
    return FakeVectorService()


class FakeParser:
    def get_chunk_objects(self, filename, file_bytes):
        return [
            Chunk(
                text="chunk about topic",
                page_no=1,
                chunk_index=0,
                content_hash="hash-0",
            )
        ]


class FakeEmbeddingService:
    def __init__(self):
        self.embedded = []

    def embed_texts(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        self.embedded.extend(texts)
        return [[0.1, 0.2] for _ in texts]


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddingService()


class FakeChatFactory:
    def __init__(self):
        self.provider_error = None

    def __call__(self, credentials):
        factory = self

        class _Provider:
            def stream_response(self, query, context):
                if factory.provider_error is not None:
                    raise factory.provider_error
                yield "The answer is 42."

        return _Provider()


@pytest.fixture
def fake_chat():
    return FakeChatFactory()


@pytest.fixture
def app(
    repositories,
    fake_storage,
    fake_vectors,
    fake_embeddings,
    fake_chat,
    settings_obj,
):
    parser = FakeParser()
    services = replace(
        Services.from_settings(settings_obj),
        repositories=repositories,
        storage=fake_storage,
        parser=parser,
        embedding_service=fake_embeddings,
        vector_service=fake_vectors,
        chat_provider_factory=fake_chat,
    )
    application = create_app(settings_obj, services=services)
    application.config.update(
        TEST_REPOSITORIES=repositories,
        TEST_STORAGE=fake_storage,
        TEST_PARSER=parser,
        TEST_EMBEDDINGS=fake_embeddings,
        TEST_VECTORS=fake_vectors,
    )
    return application


@pytest.fixture
def client(app):
    with app.test_client() as client:
        yield client


def upload(client, name="doc.pdf"):
    data = {"file": (io.BytesIO(b"%PDF-fake-bytes"), name)}
    return client.post("/upload", data=data, content_type="multipart/form-data")


def _make_pdf() -> bytes:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Backfill Heading")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"response": "OK"}


def test_file_meta_missing_returns_404(client):
    response = client.get("/files/no-such-doc.pdf/meta")

    assert response.status_code == 404
    assert response.get_json() == {"error": "File not found"}


@pytest.mark.parametrize(
    "url",
    [
        "/files/../evil.pdf/meta",
        "/files/..%2Fevil.pdf/meta",
    ],
)
def test_file_meta_traversal_rejected(client, url):
    response = client.get(url)

    assert response.status_code == 400
    assert response.get_json() == {"error": "Invalid filename"}


def test_storage_download_missing_returns_404(client):
    response = client.get("/storage/no-such-doc.pdf")

    assert response.status_code == 404
    assert response.get_json() == {"error": "File not found"}


@pytest.mark.parametrize(
    "url",
    [
        "/storage/../evil.pdf",
        "/storage/..%2Fevil.pdf",
    ],
)
def test_storage_download_traversal_rejected(client, url):
    response = client.get(url)

    assert response.status_code == 400
    assert response.get_json() == {"error": "Invalid filename"}


@pytest.mark.parametrize(
    "filename",
    # Backslash is an ordinary filename character on POSIX, so only
    # forward-slash and absolute forms can escape the storage root here.
    ["../evil.pdf", "/abs/path.pdf", "", "a/../../b.pdf"],
)
def test_storage_path_rejects_traversal(tmp_path, filename):
    storage = LocalStorage(tmp_path / "uploads")

    with pytest.raises(ValueError, match="Invalid storage filename"):
        storage._path(filename)


def test_storage_path_accepts_plain_names(tmp_path):
    storage = LocalStorage(tmp_path / "uploads")

    assert storage._path("doc.pdf").name == "doc.pdf"


def test_process_retry_after_embed_failure_succeeds(
    client, app, fake_vectors, fake_embeddings
):
    filename = upload(client).get_json()["file"]["name"]

    def _boom(texts):
        raise RuntimeError("embed down")

    fake_embeddings.embed_texts = _boom
    assert client.post("/process-file", json={"filename": filename}).status_code == 202
    assert build_test_worker(app).drain() == 1
    assert (
        client.post("/file/is-processed", json={"filename": filename}).get_json()[
            "is_processed"
        ]
        is False
    )
    deletes_after_failure = list(fake_vectors.deleted)

    fake_embeddings.embed_texts = FakeEmbeddingService().embed_texts
    retry = client.post(f"/ingestion-jobs/{filename}/retry")
    assert retry.status_code == 201
    assert build_test_worker(app).drain() == 1
    # The retry cleaned stale vectors before writing again.
    assert len(fake_vectors.deleted) > len(deletes_after_failure)
    assert fake_vectors.upserts and fake_vectors.upserts[0][2] == filename
    assert (
        client.post("/file/is-processed", json={"filename": filename}).get_json()[
            "is_processed"
        ]
        is True
    )


def test_sse_error_path_yields_error_and_empty_sources(client, app, fake_chat):
    filename = upload(client).get_json()["file"]["name"]
    client.post("/process-file", json={"filename": filename})
    build_test_worker(app).drain()
    fake_chat.provider_error = RuntimeError("provider down")

    response = client.post("/response", json={"query": "what?", "filename": filename})

    assert response.status_code == 200
    assert response.mimetype == "text/event-stream"
    events = parse_sse(response.get_data(as_text=True))
    by_name = {name: data for name, data in events}
    assert "error" in by_name
    assert by_name["error"]["error"]
    assert by_name["done"] == {
        "done": True,
        "sources": [],
        "retrieval": {"method": "dense", "outcome": "success"},
    }

    history = client.get(f"/messages?filename={filename}").get_json()["messages"]
    assert [m["sender"] for m in history] == ["user", "bot"]
    assert history[-1]["text"] == by_name["error"]["error"]


def test_mark_opened_tracks_recents(client):
    filename = upload(client).get_json()["file"]["name"]

    listing = client.get("/files").get_json()["files"]
    entry = next(f for f in listing if f["name"] == filename)
    assert entry["last_opened_at"] is None

    assert client.post("/file/opened", json={}).status_code == 400
    assert client.post("/file/opened", json={"filename": "nope.pdf"}).status_code == 404
    assert client.post(
        "/file/opened", json={"filename": "../evil.pdf"}
    ).status_code in (400, 403, 404)

    response = client.post("/file/opened", json={"filename": filename})
    assert response.status_code == 200
    assert response.get_json()["last_opened_at"]

    listing = client.get("/files").get_json()["files"]
    entry = next(f for f in listing if f["name"] == filename)
    assert entry["last_opened_at"]


def test_backfilled_title_serves_meta(client, fake_storage, repositories):
    pdf = _make_pdf()
    hex_name = "ab12cd34" * 4 + ".pdf"
    with repositories.files._session_factory() as session, session.begin():
        session.add(FileRecord(filename=hex_name, title=None, original_filename=None))
    fake_storage.save(hex_name, pdf)

    listing = client.get("/files").get_json()["files"]
    entry = next(f for f in listing if f["name"] == hex_name)
    assert entry["title"] == "Backfill Heading"

    meta = client.get(f"/files/{hex_name}/meta").get_json()
    assert meta["pageCount"] == 1
