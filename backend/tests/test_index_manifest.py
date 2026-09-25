"""
Index staleness tests at the HTTP boundary.

Every indexed document records the settings its vectors were built with. When
the running configuration no longer matches, the document is stale: its
vectors are kept, chat refuses with a reindex action, and a reindex request
queues an ordinary ingestion job that only replaces the active generation
once the replacement validates.
"""

import hashlib
import io
from dataclasses import replace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import create_app
from composition import Services
from db import Base, FileRecord
from repositories import build_repositories
from services.accounts.secrets_service import encrypt_api_key
from services.indexing import manifest as manifest_module
from services.parsing.document_parser import Chunk
from services.retrieval import qdrant_store
from services.retrieval.base import RetrievalResult
from services.retrieval.hybrid import build_sparse_vector
from services.retrieval.vector_service import FETCH_K, shape_sources
from tests.ingestion_helpers import build_test_worker
from tests.sse import parse_sse

PDF_BYTES = b"%PDF-fake-bytes"


@pytest.fixture
def session_factory():
    """Create the schema in one in-memory database and return its sessions."""
    engine = create_engine(
        "sqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


@pytest.fixture
def repositories(session_factory):
    """Build all aggregate repositories over the in-memory database."""
    repos = build_repositories(session_factory)
    repos.app_settings.upsert_app_settings(
        "groq", "openai/gpt-oss-120b", encrypt_api_key("sk-test-chat-key")
    )
    return repos


class FakeStorage:
    """In-memory stand-in for the storage layer."""

    def __init__(self):
        """Initialize."""
        self.blobs = {}

    def save(self, filename, content):
        """Do save."""
        self.blobs[filename] = content

    def open(self, filename):
        """Do open."""
        return self.blobs[filename]

    def exists(self, filename):
        """Do exists."""
        return filename in self.blobs

    def delete(self, filename):
        """Do delete."""
        self.blobs.pop(filename, None)

    def list(self):
        """Do list."""
        return [
            {"name": name, "size": len(data)}
            for name, data in sorted(self.blobs.items())
        ]

    def url(self, filename):
        """Do url."""
        return f"/storage/{filename}"


class FakeVectorService:
    """Vector service that records every write and deletion."""

    def __init__(self):
        """Initialize."""
        self.upserts = []
        self.deleted = []
        self.deleted_generations = []
        self.matches = [
            {
                "content": "chunk about topic",
                "document": "doc.pdf",
                "chunk_index": 0,
                "score": 0.92,
            }
        ]

    def upsert_chunks(self, embeddings, chunks, filename, **kwargs):
        """Do upsert chunks."""
        self.upserts.append((embeddings, chunks, filename, kwargs.get("generation")))

    def query_vectors(
        self, embedding, filename, top_k=FETCH_K, query_text=None, **kwargs
    ):
        """Return the recorded matches shaped like production retrieval."""
        sources = shape_sources(self.matches)
        method = (
            "hybrid"
            if query_text and build_sparse_vector(query_text)["indices"]
            else "dense"
        )
        return RetrievalResult(
            sources=sources,
            method=method,
            outcome="success" if sources else "empty",
        )

    def delete_by_filename(self, filename):
        """Do delete by filename."""
        self.deleted.append(filename)

    def delete_by_generation(self, filename, generation):
        """Delete one generation."""
        self.deleted_generations.append((filename, generation))

    def delete_all(self):
        """Do delete all."""


class FakeParser:
    """Fixed-chunk stand-in for the parse-and-chunk pipeline."""

    def get_chunk_objects(self, filename, file_bytes):
        """Do get chunk objects."""
        return [
            Chunk(
                text="chunk about topic",
                page_no=1,
                chunk_index=0,
                content_hash="hash-0",
            ),
        ]


class FailingParser(FakeParser):
    """Parser that fails, so a replacement index never becomes active."""

    def get_chunk_objects(self, filename, file_bytes):
        """Fail before returning chunks."""
        raise RuntimeError("parser is unavailable")


class FakeEmbeddingService:
    """Fixed-vector stand-in for the embedding service."""

    def __init__(self):
        """Initialize."""
        self.embedded = []

    def embed_texts(self, texts):
        """Do embed texts."""
        if isinstance(texts, str):
            texts = [texts]
        self.embedded.extend(texts)
        return [[0.1, 0.2] for _ in texts]


class FakeChatFactory:
    """Chat provider factory that records the streams it serves."""

    def __init__(self):
        """Initialize."""
        self.streamed = []

    def __call__(self, credentials):
        """Build a provider streaming a fixed answer."""
        factory = self

        class Provider:
            def stream_response(self, query, context, history=""):
                factory.streamed.append((query, context))
                yield "The answer is 42."

        return Provider()


@pytest.fixture
def app(repositories, settings_obj):
    """Compose the app like production, with fakes wired through the factory."""
    storage = FakeStorage()
    vectors = FakeVectorService()
    services = replace(
        Services.from_settings(settings_obj),
        repositories=repositories,
        storage=storage,
        parser=FakeParser(),
        embedding_service=FakeEmbeddingService(),
        vector_service=vectors,
        chat_provider_factory=FakeChatFactory(),
    )
    application = create_app(settings_obj, services=services)
    application.config.update(
        TEST_REPOSITORIES=repositories,
        TEST_STORAGE=storage,
        TEST_PARSER=FakeParser(),
        TEST_EMBEDDINGS=FakeEmbeddingService(),
        TEST_VECTORS=vectors,
    )
    return application


@pytest.fixture
def client(app):
    """Do client."""
    with app.test_client() as client:
        yield client


def index_document(client, app, name="doc.pdf"):
    """Upload a document and run its ingestion job to ready."""
    data = {"file": (io.BytesIO(PDF_BYTES), name)}
    filename = client.post(
        "/upload", data=data, content_type="multipart/form-data"
    ).get_json()["file"]["name"]
    build_test_worker(app).drain()
    return filename


def index_status(client, filename):
    """Read the index status the interface sees for one document."""
    return client.post("/file/is-processed", json={"filename": filename}).get_json()[
        "index"
    ]


def ask(client, filename, query="What is the answer?"):
    """Ask a question and return the parsed SSE events."""
    response = client.post("/response", json={"query": query, "filename": filename})
    return response


# Every setting whose change makes stored vectors incompatible with the
# running application, applied to the pinned test settings or the retrieval
# constants the manifest reads.
SETTING_CHANGES = [
    ("parsing.use_docling", "true", "parser"),
    ("manifest.PARSER_VERSION", "pymupdf-v0", "parser_version"),
    ("chunking.chunk_size_tokens", 1024, "chunk_size_tokens"),
    ("chunking.chunk_overlap_tokens", 128, "chunk_overlap_tokens"),
    ("embedding.embedding_model", "BAAI/bge-small-en-v1.5", "embedding_model"),
    ("embedding.revision", "9f1c2ab", "embedding_revision"),
    ("qdrant.DENSE_SIZE", 512, "vector_dimension"),
    ("manifest.SPARSE_METHOD", "hashed-tf-qdrant-idf-v2", "sparse_method"),
    (
        "manifest.TOKENIZER_VERSION",
        "lowercase-regex-stopwords-v2",
        "sparse_tokenizer_version",
    ),
    ("rerank.rerank_model", "BAAI/bge-reranker-v2-m3", "reranker_model"),
    ("rerank.revision", "77b0de1", "reranker_revision"),
    ("vector.index_name", "other-index", "collection_name"),
    (
        "manifest.COLLECTION_SCHEMA_VERSION",
        "named-dense-sparse-v2",
        "collection_schema_version",
    ),
]
SETTING_CHANGE_IDS = [change[2] for change in SETTING_CHANGES]


def apply_change(monkeypatch, settings_obj, target, value):
    """Change one runtime setting the manifest is built from."""
    group, _, field = target.partition(".")
    if group == "manifest":
        monkeypatch.setattr(manifest_module, field, value)
    elif group == "qdrant":
        monkeypatch.setattr(qdrant_store, field, value)
    else:
        monkeypatch.setattr(getattr(settings_obj, group), field, value)


def test_indexed_document_records_every_retrieval_setting(client, app, repositories):
    """A ready document exposes the manifest its vectors were built with."""
    filename = index_document(client, app)

    index = index_status(client, filename)
    assert index["state"] == "ready"
    assert index["changes"] == []
    assert index["change_details"] == []
    assert index["runtime_manifest"] == index["manifest"]

    manifest = index["manifest"]
    assert manifest["parser"] == "auto"
    assert manifest["chunk_size_tokens"] == 512
    assert manifest["chunk_overlap_tokens"] == 50
    assert manifest["embedding_model"] == "BAAI/bge-m3"
    assert manifest["vector_dimension"] == 1024
    assert manifest["sparse_method"] == manifest_module.SPARSE_METHOD
    assert manifest["sparse_tokenizer_version"] == manifest_module.TOKENIZER_VERSION
    assert manifest["collection_name"] == "pdf-index"
    assert manifest["collection_schema_version"] == (
        manifest_module.COLLECTION_SCHEMA_VERSION
    )
    assert manifest["index_generation"] == 1
    assert manifest["content_hash"] == hashlib.sha256(PDF_BYTES).hexdigest()

    listing = client.get("/files").get_json()["files"][0]
    assert listing["index"]["state"] == "ready"
    assert listing["index"]["manifest"] == manifest
    assert repositories.files.get_file(filename)["index_stale_reason"] is None


def test_matching_manifest_keeps_serving_chat(client, app):
    """A document whose manifest matches still answers questions."""
    filename = index_document(client, app)

    response = ask(client, filename)
    assert response.status_code == 200
    events = parse_sse(response.get_data(as_text=True))
    assert events[-1][1]["done"] is True
    assert index_status(client, filename)["state"] == "ready"


def test_document_without_an_index_is_pending_not_stale(client, app):
    """A document that was never indexed reports progress, not staleness."""
    data = {"file": (io.BytesIO(PDF_BYTES), "doc.pdf")}
    filename = client.post(
        "/upload", data=data, content_type="multipart/form-data"
    ).get_json()["file"]["name"]

    index = index_status(client, filename)
    assert index["state"] == "pending"
    assert index["changes"] == []


def test_chat_refuses_a_document_with_nothing_indexed_yet(client, app):
    """An unready document is refused with the same reindex action."""
    data = {"file": (io.BytesIO(PDF_BYTES), "doc.pdf")}
    filename = client.post(
        "/upload", data=data, content_type="multipart/form-data"
    ).get_json()["file"]["name"]

    response = ask(client, filename)

    assert response.status_code == 409
    body = response.get_json()
    assert body["category"] == "index_pending"
    assert body["action"] == "reindex"
    assert body["index"]["state"] == "pending"


def test_document_indexed_before_manifests_are_stale(client, app, session_factory):
    """A processed document with no recorded manifest needs one reindex."""
    filename = index_document(client, app)
    with session_factory() as session, session.begin():
        record = session.scalars(
            select(FileRecord).where(FileRecord.filename == filename)
        ).first()
        record.index_manifest = None

    index = index_status(client, filename)
    assert index["state"] == "stale"
    assert [d["label"] for d in index["change_details"]] == ["index manifest"]
    assert index["manifest"] is None


def test_toggling_the_reranker_keeps_the_document_ready(
    client, app, settings_obj, monkeypatch
):
    """The reranker gate runs at query time, so it forces no reindex."""
    filename = index_document(client, app)

    monkeypatch.setattr(settings_obj.rerank, "enabled", True)

    index = index_status(client, filename)
    assert index["state"] == "ready"
    assert index["runtime_manifest"]["reranker_enabled"] is True
    assert ask(client, filename).status_code == 200


@pytest.mark.parametrize(
    ("target", "value", "field"),
    SETTING_CHANGES,
    ids=SETTING_CHANGE_IDS,
)
def test_each_retrieval_setting_change_marks_the_document_stale(
    client, app, repositories, settings_obj, monkeypatch, target, value, field
):
    """A parser, chunking, model, or collection change makes the index stale."""
    filename = index_document(client, app)
    vectors = app.config["TEST_VECTORS"]

    apply_change(monkeypatch, settings_obj, target, value)

    label = manifest_module.CHANGE_LABELS[field]
    index = index_status(client, filename)
    assert index["state"] == "stale", field
    assert field in index["changes"], field
    assert [d["label"] for d in index["change_details"]] == [label], field
    # The old vectors stay: staleness must never delete a usable index.
    assert vectors.deleted == []
    assert vectors.deleted_generations == []
    assert repositories.files.get_file(filename)["is_processed"] is True
    assert label in repositories.files.get_file(filename)["index_stale_reason"]


@pytest.mark.parametrize(
    ("target", "value", "field"),
    SETTING_CHANGES,
    ids=SETTING_CHANGE_IDS,
)
def test_chat_refuses_a_stale_document_with_a_reindex_action(
    client, app, settings_obj, monkeypatch, target, value, field
):
    """Chat stops before querying an index the app no longer serves."""
    filename = index_document(client, app)
    apply_change(monkeypatch, settings_obj, target, value)

    response = ask(client, filename)

    assert response.status_code == 409
    body = response.get_json()
    assert body["category"] == "index_stale"
    assert body["action"] == "reindex"
    detail = body["index"]["change_details"][0]
    assert detail["label"] == manifest_module.CHANGE_LABELS[field]
    assert detail["indexed"] != detail["current"]
    assert body["index"]["manifest"] is not None
    assert body["index"]["runtime_manifest"] is not None
    # The refusal happens before retrieval, so no answer was streamed.
    assert "streamed" not in body


def test_reindex_queues_a_job_and_keeps_the_active_generation_until_it_succeeds(
    client, app, settings_obj, monkeypatch
):
    """A reindex is an ordinary ingestion job writing a new generation."""
    filename = index_document(client, app)
    vectors = app.config["TEST_VECTORS"]
    monkeypatch.setattr(settings_obj.chunking, "chunk_size_tokens", 1024)

    response = client.post(f"/files/{filename}/reindex")
    assert response.status_code == 201
    job = response.get_json()["job"]
    assert job["state"] == "queued"
    assert job["generation"] == 2
    assert job["filename"] == filename

    # Nothing was deleted and no replacement vectors exist yet.
    assert vectors.upserts[-1][3] == 1
    assert vectors.deleted_generations == []
    record = app.config["TEST_REPOSITORIES"].files.get_file(filename)
    assert record["is_processed"] is True
    assert record["index_generation"] == 1

    build_test_worker(app).drain()

    assert vectors.upserts[-1][3] == 2
    assert vectors.deleted_generations == [(filename, 1)]
    assert index_status(client, filename)["state"] == "ready"
    record = app.config["TEST_REPOSITORIES"].files.get_file(filename)
    assert record["index_generation"] == 2
    assert record["index_stale_reason"] is None
    assert ask(client, filename).status_code == 200


def test_failed_reindex_keeps_the_previous_index_serving_chat(
    client, app, settings_obj, monkeypatch
):
    """A replacement that never validates leaves the old index in place."""
    filename = index_document(client, app)
    vectors = app.config["TEST_VECTORS"]
    original_manifest = index_status(client, filename)["manifest"]
    monkeypatch.setattr(settings_obj.chunking, "chunk_size_tokens", 1024)
    assert client.post(f"/files/{filename}/reindex").status_code == 201

    app.config["TEST_PARSER"] = FailingParser()
    build_test_worker(app).drain()

    job = app.config["TEST_REPOSITORIES"].ingestion_jobs.get_latest(filename)
    assert job["state"] == "failed"
    record = app.config["TEST_REPOSITORIES"].files.get_file(filename)
    assert record["is_processed"] is True
    assert record["index_generation"] == 1
    # The failed attempt's own partial write was cleaned up; generation 1, the
    # index the user was reading, was never touched.
    assert vectors.upserts[-1][3] == 1
    assert vectors.deleted_generations == [(filename, 2)]

    monkeypatch.setattr(settings_obj.chunking, "chunk_size_tokens", 512)
    assert index_status(client, filename)["state"] == "ready"
    assert index_status(client, filename)["manifest"] == original_manifest
    assert ask(client, filename).status_code == 200


def test_reindex_reports_a_running_reindex_instead_of_queueing_a_second(
    client, app, settings_obj, monkeypatch
):
    """A second reindex request reuses the job already in flight."""
    filename = index_document(client, app)
    monkeypatch.setattr(settings_obj.chunking, "chunk_size_tokens", 1024)

    first = client.post(f"/files/{filename}/reindex").get_json()["job"]
    second = client.post(f"/files/{filename}/reindex")

    assert second.status_code == 202
    assert second.get_json()["job"]["id"] == first["id"]


def test_reindex_rejects_a_deleting_or_missing_document(client, app):
    """Reindex never resurrects a document that is going away."""
    filename = index_document(client, app)
    assert client.post("/files/does-not-exist.pdf/reindex").status_code == 404

    app.config["TEST_REPOSITORIES"].files.mark_deleting(filename)
    response = client.post(f"/files/{filename}/reindex")
    assert response.status_code == 409
    assert response.get_json()["category"] == "document_deleting"
