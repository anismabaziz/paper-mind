"""
Durable deletion tests at the HTTP boundary.

Uses the same fakes as test_flows: in-memory storage, fake vectors, and
in-memory SQLite. Asserts user-visible behavior only: status codes,
payloads, stored rows, and remaining bytes/vectors.
"""

import io
from dataclasses import replace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import create_app
from composition import Services
from db import Base, Message, Source
from repositories import build_repositories
from services.accounts.secrets_service import encrypt_api_key
from services.parsing.document_parser import Chunk
from services.retrieval.base import RetrievalResult
from services.retrieval.hybrid import build_sparse_vector
from services.retrieval.vector_service import FETCH_K
from tests.sse import parse_sse


@pytest.fixture
def repositories():
    """Build all aggregate repositories over one in-memory database."""
    engine = create_engine(
        "sqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    repos = build_repositories(sessionmaker(bind=engine))
    repos.app_settings.upsert_app_settings(
        "groq", "openai/gpt-oss-120b", encrypt_api_key("sk-test-chat-key")
    )
    return repos


class FakeStorage:
    """In-memory stand-in for the storage layer."""

    def __init__(self):
        """Initialize."""
        self.blobs = {}
        self.fail_delete = False

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
        if self.fail_delete:
            raise RuntimeError("disk down")
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


@pytest.fixture
def fake_storage():
    """Fresh in-memory storage, injected through the app factory."""
    return FakeStorage()


class FakeVectorService:
    """FakeVectorService."""

    def __init__(self):
        """Initialize."""
        self.upserts = []
        self.deleted = []
        self.fail_delete = False
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
        self.upserts.append((embeddings, chunks, filename))

    def query_vectors(
        self, embedding, filename, top_k=FETCH_K, query_text=None, **kwargs
    ):
        """Do query vectors."""
        from services.retrieval.vector_service import shape_sources

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
        if self.fail_delete:
            raise RuntimeError("qdrant down")
        self.deleted.append(filename)

    def delete_all(self):
        """Do delete all."""
        pass


@pytest.fixture
def fake_vectors():
    """Fresh fake vector service, injected through the app factory."""
    return FakeVectorService()


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


class FakeEmbeddingService:
    """Fixed-vector stand-in that records every embedded text."""

    def __init__(self):
        """Initialize."""
        self.embedded = []

    def embed_texts(self, texts):
        """Do embed texts."""
        if isinstance(texts, str):
            texts = [texts]
        self.embedded.extend(texts)
        return [[0.1, 0.2] for _ in texts]


class FakeChatProvider:
    """Streams fixed tokens through its factory so tests can observe calls."""

    def __init__(self, factory):
        """Initialize."""
        self._factory = factory

    def stream_response(self, query, context):
        """Do stream response."""
        yield "The answer "
        yield "is 42."

    def generate_response(self, query, context):
        """Do generate response."""
        return "The answer is 42."


class FakeChatFactory:
    """Fake chat provider factory."""

    def __init__(self):
        """Initialize."""
        self.built = None

    def __call__(self, credentials):
        """Do build provider."""
        self.built = credentials
        return FakeChatProvider(self)


@pytest.fixture
def app(
    repositories,
    fake_storage,
    fake_vectors,
    settings_obj,
):
    """Compose the app like production, with fakes wired through the factory."""
    services = replace(
        Services.from_settings(settings_obj),
        repositories=repositories,
        storage=fake_storage,
        parser=FakeParser(),
        embedding_service=FakeEmbeddingService(),
        vector_service=fake_vectors,
        chat_provider_factory=FakeChatFactory(),
    )
    return create_app(settings_obj, services=services)


@pytest.fixture
def client(app):
    """Do client."""
    with app.test_client() as client:
        yield client


def upload(client, name="doc.pdf"):
    """Do upload."""
    data = {"file": (io.BytesIO(b"%PDF-fake-bytes"), name)}
    return client.post("/upload", data=data, content_type="multipart/form-data")


def upload_process_chat(client):
    """Upload, index, and ask once so deletion has vectors and history."""
    filename = upload(client).get_json()["file"]["name"]
    assert client.post("/process-file", json={"filename": filename}).status_code == 200
    streamed = client.post("/response", json={"query": "what?", "filename": filename})
    assert streamed.status_code == 200
    # Consume the SSE stream so the bot reply is persisted before assertions.
    streamed.get_data(as_text=True)
    return filename


def test_vector_failure_keeps_retryable_delete_failed_state(
    client, fake_storage, fake_vectors, repositories
):
    """A vector failure never reports success and retains state for retry."""
    filename = upload_process_chat(client)
    fake_vectors.fail_delete = True

    response = client.delete(f"/files/remove?path={filename}")

    assert response.status_code == 500
    assert response.get_json()["category"] == "document_delete_failed"
    assert "vectors" in response.get_json()["leftovers"]
    record = repositories.files.get_file(filename)
    assert record["deletion_state"] == "delete_failed"
    assert record["deletion_error"]
    assert record["deletion_attempts"] >= 1
    # Metadata is retained so the failed cleanup can be retried; the
    # response never claims success while vectors may remain.
    assert repositories.files.list_files() != []

    fake_vectors.fail_delete = False
    retry = client.delete(f"/files/remove?path={filename}")

    assert retry.status_code == 200
    assert fake_storage.blobs == {}
    assert repositories.files.list_files() == []
    assert client.get(f"/messages?filename={filename}").get_json()["messages"] == []
    assert filename in fake_vectors.deleted


def test_storage_failure_keeps_bytes_and_metadata_for_retry(
    client, fake_storage, fake_vectors, repositories
):
    """A disk failure during cleanup is a 500 with the document retained."""
    filename = upload_process_chat(client)
    fake_storage.fail_delete = True

    response = client.delete(f"/files/remove?path={filename}")

    assert response.status_code == 500
    assert "file" in response.get_json()["leftovers"]
    record = repositories.files.get_file(filename)
    assert record["deletion_state"] == "delete_failed"
    assert filename in fake_storage.blobs

    fake_storage.fail_delete = False
    assert client.delete(f"/files/remove?path={filename}").status_code == 200
    assert fake_storage.blobs == {}
    assert repositories.files.list_files() == []


def test_conversation_failure_retains_history_for_retry(
    client, fake_vectors, repositories, monkeypatch
):
    """A Postgres failure deleting history blocks metadata finalization."""
    filename = upload_process_chat(client)

    def _boom(conversation_id):
        raise RuntimeError("postgres down")

    monkeypatch.setattr(
        repositories.conversations, "delete_conversation_tree", _boom
    )

    response = client.delete(f"/files/remove?path={filename}")

    assert response.status_code == 500
    assert "conversation" in response.get_json()["leftovers"]
    assert repositories.files.get_file(filename)["deletion_state"] == "delete_failed"
    assert client.get(f"/messages?filename={filename}").get_json()["messages"] != []


def test_deletion_removes_citation_sources_without_orphans(
    client, repositories
):
    """Conversation cleanup removes messages and their citation sources."""
    from db import Source as SourceModel

    filename = upload_process_chat(client)
    file_record = repositories.files.get_file(filename)
    conversation_id = repositories.conversations.get_conversation_id(
        file_record["id"]
    )
    with repositories.conversations._session_factory() as session:
        assert session.query(Message).count() == 2
        assert session.query(Source).count() >= 1
        assert session.query(SourceModel).count() >= 1

    assert client.delete(f"/files/remove?path={filename}").status_code == 200

    with repositories.conversations._session_factory() as session:
        assert session.query(Message).count() == 0
        assert session.query(Source).count() == 0


def test_chat_and_process_are_blocked_while_deleting(
    client, fake_vectors, repositories
):
    """The deleting state blocks new chat and ingestion with a 409."""
    filename = upload_process_chat(client)
    fake_vectors.fail_delete = True
    assert client.delete(f"/files/remove?path={filename}").status_code == 500
    fake_vectors.fail_delete = False

    chat = client.post("/response", json={"query": "what?", "filename": filename})
    assert chat.status_code == 409
    assert chat.get_json()["category"] == "document_delete_failed"

    process = client.post("/process-file", json={"filename": filename})
    assert process.status_code == 409
    assert process.get_json()["category"] == "document_delete_failed"

    # The failed deletion itself remains retryable after blocked attempts.
    assert client.delete(f"/files/remove?path={filename}").status_code == 200


def test_listing_exposes_deletion_state_for_retry_ui(
    client, fake_vectors, repositories
):
    """The file listing carries the failed-deletion state the UI renders."""
    filename = upload_process_chat(client)
    assert client.get("/files").get_json()["files"][0]["deletion_state"] == "active"

    fake_vectors.fail_delete = True
    assert client.delete(f"/files/remove?path={filename}").status_code == 500

    files = client.get("/files").get_json()["files"]
    assert files[0]["deletion_state"] == "delete_failed"
    assert files[0]["deletion_error"]
    assert files[0]["deletion_attempts"] >= 1


def test_concurrent_deletes_serialize_to_one_clean_final_state(
    app, client, fake_storage, fake_vectors, repositories
):
    """Two simultaneous deletions cannot leave or recreate document data."""
    import threading

    filename = upload_process_chat(client)
    results = []
    results_lock = threading.Lock()

    def _delete():
        with app.test_client() as thread_client:
            status = thread_client.delete(f"/files/remove?path={filename}").status_code
        with results_lock:
            results.append(status)

    threads = [threading.Thread(target=_delete) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert sorted(results) == [200, 200]
    assert fake_storage.blobs == {}
    assert repositories.files.list_files() == []
    assert client.get(f"/messages?filename={filename}").get_json()["messages"] == []
    assert client.get("/files").get_json()["files"] == []
