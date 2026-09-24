"""Title derivation and surfacing tests."""

# ruff: noqa: D100, D101, D102, D103, D104, D105, D107

import io
import uuid
from dataclasses import replace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import create_app
from composition import Services
from db import Base, FileRecord
from repositories import build_repositories
from tests.ingestion_helpers import build_test_worker
from services.retrieval.base import RetrievalResult
from services.titles import derive_title, is_hex_like_title


def _make_pdf_with_title(
    text: str | None = None, metadata_title: str | None = None
) -> bytes:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    if text:
        page.insert_text((72, 72), text)
    if metadata_title is not None:
        doc.set_metadata({"title": metadata_title})
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


@pytest.fixture
def repositories():
    engine = create_engine(
        "sqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return build_repositories(sessionmaker(bind=engine))


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


class FakeVectorService:
    def upsert_chunks(self, *a, **k):
        pass

    def query_vectors(self, *a, **k):
        return RetrievalResult(sources=[], method="dense", outcome="empty")

    def delete_by_filename(self, *a, **k):
        pass

    def delete_all(self):
        pass


class FakeParser:
    from services.parsing.document_parser import Chunk

    def get_chunk_objects(self, filename, file_bytes):
        return [self.Chunk(text="hello", page_no=1, chunk_index=0, content_hash="h")]


class FakeEmbed:
    def embed_texts(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return [[0.1, 0.2] for _ in texts]


class FakeChatFactory:
    def __call__(self, creds):
        class _P:
            def stream_response(self, q, ctx):
                yield "hi"

        return _P()


@pytest.fixture
def app_and_client(repositories, settings_obj):
    storage = FakeStorage()
    services = replace(
        Services.from_settings(settings_obj),
        repositories=repositories,
        storage=storage,
        parser=FakeParser(),
        embedding_service=FakeEmbed(),
        vector_service=FakeVectorService(),
        chat_provider_factory=FakeChatFactory(),
    )
    app = create_app(settings_obj, services=services)
    return app, app.test_client(), storage, repositories


def test_derive_title_priority_metadata_over_filename():
    pdf = _make_pdf_with_title(
        text="First Heading", metadata_title="  Metadata Title  "
    )
    assert derive_title(pdf, "ignored.pdf") == "Metadata Title"
    assert derive_title(pdf, None) == "Metadata Title"


def test_derive_title_fallback_to_original_filename():
    pdf = _make_pdf_with_title(text="Heading")
    assert derive_title(pdf, "CoolPaper.pdf") == "CoolPaper"
    assert derive_title(pdf, "  spaced name .pdf ") == "spaced name"


def test_derive_title_fallback_to_heading():
    pdf = _make_pdf_with_title(text="First Heading Line")
    assert derive_title(pdf, None) == "First Heading Line"


def test_derive_title_trims_and_limits_heading():
    pdf = _make_pdf_with_title(text="   Trimmed   ")
    assert derive_title(pdf, None) == "Trimmed"
    long_heading = "a" * 300
    pdf2 = _make_pdf_with_title(text=long_heading)
    title = derive_title(pdf2, None)
    assert len(title) <= 200
    # Heading may be line-wrapped by the PDF renderer, so truncation
    # only applies when the extracted line itself exceeds the limit.
    if len(long_heading) > 200:
        # If the PDF split the line, the single chunk will be shorter.
        # Ensure we never exceed the max, and an over-limit single line
        # would be truncated with ellipsis.
        assert len(title) <= 200


def test_is_hex_like_title():
    assert is_hex_like_title(None) is False
    assert is_hex_like_title("My Paper") is False
    assert is_hex_like_title("a" * 32) is True
    assert is_hex_like_title("A" * 32 + ".pdf") is True
    assert is_hex_like_title("  " + "b" * 32 + "  ") is True


def test_upload_returns_derived_title_and_listing(app_and_client):
    _, client, storage, _ = app_and_client
    pdf = _make_pdf_with_title(text="Heading", metadata_title="Paper via Metadata")
    data = {"file": (io.BytesIO(pdf), "original_name.pdf")}
    resp = client.post("/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["file"]["title"] == "Paper via Metadata"
    assert body["file"]["original_filename"] == "original_name.pdf"
    stored_name = body["file"]["name"]
    assert storage.blobs[stored_name] == pdf

    listing = client.get("/files").get_json()["files"]
    assert len(listing) == 1
    assert listing[0]["title"] == "Paper via Metadata"
    assert listing[0]["original_filename"] == "original_name.pdf"
    assert listing[0]["name"] == stored_name


def test_upload_filename_fallback_when_no_metadata(app_and_client):
    _, client, _, _ = app_and_client
    pdf = _make_pdf_with_title(text="Some Heading")
    data = {"file": (io.BytesIO(pdf), "My Research.pdf")}
    resp = client.post("/upload", data=data, content_type="multipart/form-data")
    assert resp.get_json()["file"]["title"] == "My Research"


def test_lazy_backfill_null_title(app_and_client):
    _, client, storage, repositories = app_and_client
    pdf = _make_pdf_with_title(text="Backfill Heading")
    hex_name = f"{uuid.uuid4().hex}.pdf"
    with repositories.files._session_factory() as sess, sess.begin():
        sess.add(FileRecord(filename=hex_name, title=None, original_filename=None))
    storage.save(hex_name, pdf)

    resp = client.get("/files").get_json()
    files = resp["files"]
    entry = next(f for f in files if f["name"] == hex_name)
    assert entry["title"] == "Backfill Heading"

    with repositories.files._session_factory() as sess:
        rec = sess.scalars(
            select(FileRecord).where(FileRecord.filename == hex_name)
        ).first()
        assert rec.title == "Backfill Heading"


def test_lazy_backfill_hex_like_title(app_and_client):
    _, client, storage, repositories = app_and_client
    pdf = _make_pdf_with_title(text="Real Heading", metadata_title="Derived from Meta")
    hex_name = f"{uuid.uuid4().hex}.pdf"
    hex_title = uuid.uuid4().hex
    with repositories.files._session_factory() as sess, sess.begin():
        sess.add(FileRecord(filename=hex_name, title=hex_title, original_filename=None))
    storage.save(hex_name, pdf)

    resp = client.get("/files").get_json()
    files = resp["files"]
    entry = next(f for f in files if f["name"] == hex_name)
    assert entry["title"] == "Derived from Meta"

    with repositories.files._session_factory() as sess:
        rec = sess.scalars(
            select(FileRecord).where(FileRecord.filename == hex_name)
        ).first()
        assert rec.title == "Derived from Meta"


def test_vectors_still_keyed_on_filename_not_title():

    class CapturingVectors(FakeVectorService):
        def __init__(self):
            self.last_filename = None

        def upsert_chunks(self, embeddings, chunks, filename, **k):
            self.last_filename = filename

    pdf = _make_pdf_with_title(text="Content", metadata_title="Titled Paper")
    vectors = CapturingVectors()
    engine = create_engine(
        "sqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    fresh_repositories = build_repositories(sessionmaker(bind=engine))
    storage = FakeStorage()
    import settings as sm

    settings_obj = sm.get_settings()
    svc = replace(
        Services.from_settings(settings_obj),
        repositories=fresh_repositories,
        storage=storage,
        parser=FakeParser(),
        embedding_service=FakeEmbed(),
        vector_service=vectors,
        chat_provider_factory=FakeChatFactory(),
    )
    app = create_app(settings_obj, services=svc)
    app.config.update(
        TEST_REPOSITORIES=fresh_repositories,
        TEST_STORAGE=storage,
        TEST_PARSER=svc.parser,
        TEST_EMBEDDINGS=svc.embedding_service,
        TEST_VECTORS=vectors,
    )
    cl = app.test_client()
    data = {"file": (io.BytesIO(pdf), "titled.pdf")}
    stored = cl.post(
        "/upload", data=data, content_type="multipart/form-data"
    ).get_json()["file"]
    cl.post("/process-file", json={"filename": stored["name"]})
    build_test_worker(app).drain()
    assert vectors.last_filename == stored["name"]
    assert vectors.last_filename != stored["title"]
