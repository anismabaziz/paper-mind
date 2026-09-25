"""Ordered conversation turn tests at the repository and HTTP boundaries."""

import io
from dataclasses import replace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import create_app
from composition import Services
from db import Base, Conversation, Source, Turn
from repositories import ConversationRepository, FileRepository, build_repositories
from services.accounts.secrets_service import encrypt_api_key
from services.parsing.document_parser import Chunk
from services.retrieval.base import RetrievalResult
from services.retrieval.vector_service import FETCH_K
from tests.ingestion_helpers import build_test_worker
from tests.sse import parse_sse

CHUNK = Chunk(text="chunk about topic", page_no=1, chunk_index=0, content_hash="hash-0")


def _in_memory_session_factory():
    """Build a session factory over one in-memory database."""
    engine = create_engine(
        "sqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


class _Storage:
    """In-memory stand-in for the storage layer."""

    def __init__(self):
        """Start with no stored bytes."""
        self.blobs = {}

    def save(self, filename, content):
        """Store bytes."""
        self.blobs[filename] = content

    def open(self, filename):
        """Read bytes."""
        return self.blobs[filename]

    def exists(self, filename):
        """Report whether bytes are stored."""
        return filename in self.blobs

    def delete(self, filename):
        """Remove bytes."""
        self.blobs.pop(filename, None)

    def list(self):
        """List stored files."""
        return [
            {"name": name, "size": len(data)}
            for name, data in sorted(self.blobs.items())
        ]

    def url(self, filename):
        """Return the download path for a stored file."""
        return f"/storage/{filename}"


class _Vectors:
    """Fixed-match stand-in for the vector service."""

    def upsert_chunks(self, embeddings, chunks, filename, **kwargs):
        """Accept an upsert without storing anything."""

    def query_vectors(
        self, embedding, filename, top_k=FETCH_K, query_text=None, **kwargs
    ):
        """Return one grounded source."""
        return RetrievalResult(
            sources=[
                {
                    "content": "chunk about topic",
                    "document": "doc.pdf",
                    "chunk_index": 0,
                    "score": 0.9,
                    "page": 1,
                }
            ],
            method="hybrid",
            outcome="success",
        )

    def delete_by_filename(self, filename):
        """Accept a delete without storing anything."""

    def delete_all(self):
        """Accept a bulk delete without storing anything."""


class _Parser:
    """Single-chunk stand-in for the parse-and-chunk pipeline."""

    def get_chunk_objects(self, filename, file_bytes):
        """Return one chunk."""
        return [CHUNK]


class _Embeddings:
    """Fixed-vector stand-in."""

    def embed_texts(self, texts):
        """Return one vector per text."""
        if isinstance(texts, str):
            texts = [texts]
        return [[0.1, 0.2] for _ in texts]


class _ChatProvider:
    """Streams fixed tokens."""

    def __init__(self, factory):
        """Bind the provider to its factory."""
        self._factory = factory

    def stream_response(self, query, context, history=""):
        """Yield a fixed answer, or raise the requested provider error."""
        if self._factory.provider_error is not None:
            raise self._factory.provider_error
        self._factory.streamed.append((query, context, history))
        yield "The answer "
        yield "is 42."


class _ChatFactory:
    """Chat provider factory that can fail the next stream."""

    def __init__(self):
        """Start without an active provider error."""
        self.streamed = []
        self.provider_error = None

    def __call__(self, credentials):
        """Build a streaming provider."""
        return _ChatProvider(self)


@pytest.fixture
def session_factory():
    """Session factory over one in-memory database."""
    return _in_memory_session_factory()


@pytest.fixture
def conversation(session_factory):
    """Create one stored document and the conversation bound to it."""
    files = FileRepository(session_factory)
    file_id = files.create_file("doc.pdf", title="Doc")["id"]
    repository = ConversationRepository(session_factory)
    return repository, repository.ensure_conversation(file_id), file_id


@pytest.fixture
def repositories(session_factory):
    """Build every repository over one in-memory database with saved settings."""
    repositories = build_repositories(session_factory)
    repositories.app_settings.upsert_app_settings(
        "groq", "openai/gpt-oss-120b", encrypt_api_key("sk-test-chat-key")
    )
    return repositories


@pytest.fixture
def fake_chat():
    """Chat provider factory that can fail the next stream."""
    return _ChatFactory()


@pytest.fixture
def app(repositories, fake_chat, settings_obj):
    """Compose the app with fakes over the in-memory database."""
    storage = _Storage()
    services = replace(
        Services.from_settings(settings_obj),
        repositories=repositories,
        storage=storage,
        parser=_Parser(),
        embedding_service=_Embeddings(),
        vector_service=_Vectors(),
        chat_provider_factory=fake_chat,
    )
    application = create_app(settings_obj, services=services)
    application.config.update(
        TEST_REPOSITORIES=repositories,
        TEST_STORAGE=storage,
        TEST_PARSER=services.parser,
        TEST_EMBEDDINGS=services.embedding_service,
        TEST_VECTORS=services.vector_service,
    )
    return application


@pytest.fixture
def client(app):
    """HTTP client for the composed app."""
    return app.test_client()


def indexed_document(client, app, name="doc.pdf") -> str:
    """Upload a document and index it so chat can be asked a question."""
    data = {"file": (io.BytesIO(b"%PDF-fake-bytes"), name)}
    filename = client.post(
        "/upload", data=data, content_type="multipart/form-data"
    ).get_json()["file"]["name"]
    assert client.post("/process-file", json={"filename": filename}).status_code == 202
    assert build_test_worker(app, "test-worker").drain() == 1
    return filename


def turns_of(repositories, filename) -> list[dict]:
    """Read one document's conversation turns in sequence order."""
    file_id = repositories.files.get_file(filename)["id"]
    conversation_id = repositories.conversations.get_conversation_id(file_id)
    return repositories.conversations.get_turns(conversation_id)


def test_ensuring_a_conversation_twice_reuses_the_first(conversation):
    """A repeated call returns the conversation already bound to the document."""
    repository, conversation_id, file_id = conversation

    assert repository.ensure_conversation(file_id) == conversation_id
    assert repository.get_conversation_id(file_id) == conversation_id


def test_a_second_conversation_for_one_document_is_rejected(session_factory):
    """The database refuses a duplicate conversation for the same document."""
    file_id = FileRepository(session_factory).create_file("doc.pdf", title="Doc")["id"]
    with session_factory() as session, session.begin():
        session.add(Conversation(file_id=file_id, id="a" * 32))

    with pytest.raises(IntegrityError):
        with session_factory() as session, session.begin():
            session.add(Conversation(file_id=file_id, id="b" * 32))

    with session_factory() as session:
        assert [row.id for row in session.scalars(select(Conversation)).all()] == [
            "a" * 32
        ]


def test_turns_receive_consecutive_sequences_in_order(conversation):
    """Sequences are gapless and follow the order the questions were asked."""
    repository, conversation_id, _ = conversation

    first = repository.start_turn(conversation_id, "First question")
    second = repository.start_turn(conversation_id, "Second question")
    repository.complete_turn(second, "Second answer")
    repository.complete_turn(first, "First answer")

    turns = repository.get_turns(conversation_id)
    assert [turn["sequence"] for turn in turns] == [1, 2]
    assert [turn["question"] for turn in turns] == ["First question", "Second question"]


def test_a_completed_turn_keeps_its_answer_and_sources(conversation):
    """Finishing a turn records the answer, the outcome, and the sources."""
    repository, conversation_id, _ = conversation

    turn_id = repository.start_turn(conversation_id, "What is a RAG pipeline?")
    repository.complete_turn(
        turn_id,
        "Five stages.",
        [
            {
                "content": "A RAG pipeline has five stages",
                "document": "doc.pdf",
                "chunk_index": 0,
                "score": 0.9,
                "page": 1,
            }
        ],
    )

    turn = repository.get_turns(conversation_id)[0]
    assert turn["sequence"] == 1
    assert turn["status"] == "answered"
    assert turn["question"] == "What is a RAG pipeline?"
    assert turn["answer"] == "Five stages."
    assert turn["started_at"] and turn["completed_at"]
    assert turn["failure_reason"] is None
    assert turn["sources"] == [
        {
            "content": "A RAG pipeline has five stages",
            "document": "doc.pdf",
            "chunk_index": 0,
            "score": 0.9,
            "page": 1,
        }
    ]


def test_a_failed_turn_records_the_failure_not_a_stranded_question(conversation):
    """A provider failure closes the turn with a readable reply."""
    repository, conversation_id, _ = conversation

    turn_id = repository.start_turn(conversation_id, "What is a RAG pipeline?")
    repository.fail_turn(turn_id, "The model is unavailable.", "provider failure")

    turn = repository.get_turns(conversation_id)[0]
    assert turn["status"] == "failed"
    assert turn["question"] == "What is a RAG pipeline?"
    assert turn["answer"] == "The model is unavailable."
    assert turn["failure_reason"] == "provider failure"
    assert turn["completed_at"]


def test_a_cancelled_turn_is_terminal_and_keeps_its_question(conversation):
    """A client that walks away leaves a recorded cancellation."""
    repository, conversation_id, _ = conversation

    turn_id = repository.start_turn(conversation_id, "What is a RAG pipeline?")
    repository.cancel_turn(turn_id, "client disconnected")

    turn = repository.get_turns(conversation_id)[0]
    assert turn["status"] == "cancelled"
    assert turn["question"] == "What is a RAG pipeline?"
    assert turn["answer"] is None
    assert turn["failure_reason"] == "client disconnected"
    assert turn["completed_at"]


def test_a_terminal_turn_is_not_overwritten_by_a_late_outcome(conversation):
    """A completion that arrives after a cancellation changes nothing."""
    repository, conversation_id, _ = conversation

    turn_id = repository.start_turn(conversation_id, "What is a RAG pipeline?")
    repository.cancel_turn(turn_id, "client disconnected")

    assert repository.complete_turn(turn_id, "Late answer.") is False
    turn = repository.get_turns(conversation_id)[0]
    assert turn["status"] == "cancelled"
    assert turn["answer"] is None


def test_messages_replay_the_accepted_turn_order_and_sources(conversation):
    """The message read model follows turn order and carries the outcome."""
    repository, conversation_id, _ = conversation

    first = repository.start_turn(conversation_id, "First question")
    repository.complete_turn(
        first,
        "First answer",
        [
            {
                "content": "Chunk",
                "document": "doc.pdf",
                "chunk_index": 0,
                "score": 0.5,
                "page": None,
            }
        ],
    )
    second = repository.start_turn(conversation_id, "Second question")
    repository.cancel_turn(second, "client disconnected")

    messages = repository.get_messages(conversation_id)
    assert [(message["sender"], message["text"]) for message in messages] == [
        ("user", "First question"),
        ("bot", "First answer"),
        ("user", "Second question"),
    ]
    assert [message["turn_sequence"] for message in messages] == [1, 1, 2]
    assert [message["turn_status"] for message in messages] == [
        "answered",
        "answered",
        "cancelled",
    ]
    assert len({message["id"] for message in messages}) == 3
    assert messages[1]["sources"][0]["content"] == "Chunk"
    assert messages[0]["sources"] == []


def test_deleting_a_conversation_removes_its_turns_and_sources(
    session_factory, conversation
):
    """Deletion leaves no turn or citation source behind."""
    repository, conversation_id, _ = conversation
    turn_id = repository.start_turn(conversation_id, "Question")
    repository.complete_turn(
        turn_id,
        "Answer",
        [
            {
                "content": "Chunk",
                "document": "doc.pdf",
                "chunk_index": 0,
                "score": 0.5,
                "page": None,
            }
        ],
    )

    repository.delete_conversation_tree(conversation_id)

    with session_factory() as session:
        assert list(session.scalars(select(Turn)).all()) == []
        assert list(session.scalars(select(Source)).all()) == []
        assert list(session.scalars(select(Conversation)).all()) == []


def test_a_provider_failure_reaches_history_as_a_failed_turn(
    client, app, fake_chat, repositories
):
    """The route closes the turn it opened when the provider fails."""
    filename = indexed_document(client, app)
    fake_chat.provider_error = RuntimeError("provider down")

    response = client.post("/response", json={"query": "what?", "filename": filename})

    assert [name for name, _ in parse_sse(response.get_data(as_text=True))] == [
        "error",
        "done",
    ]
    turn = turns_of(repositories, filename)[0]
    assert turn["status"] == "failed"
    assert turn["answer"]
    assert turn["failure_reason"] == "provider failure"
    history = client.get(f"/messages?filename={filename}").get_json()["messages"]
    assert [message["turn_status"] for message in history] == ["failed", "failed"]


def test_an_answer_that_cannot_be_saved_closes_the_turn_as_failed(
    client, app, repositories, monkeypatch
):
    """A persistence failure records a failure instead of an open question."""
    filename = indexed_document(client, app)

    def _boom(*args, **kwargs):
        raise RuntimeError("postgres down")

    monkeypatch.setattr(repositories.conversations, "complete_turn", _boom)

    response = client.post("/response", json={"query": "what?", "filename": filename})
    events = parse_sse(response.get_data(as_text=True))

    assert [name for name, _ in events] == ["token", "token", "error", "done"]
    turn = turns_of(repositories, filename)[0]
    assert turn["status"] == "failed"
    assert turn["failure_reason"] == "answer could not be saved"
    assert turn["answer"] in events[-2][1]["error"]


def test_a_disconnected_client_records_a_cancelled_turn(client, app, repositories):
    """Closing the stream mid-answer leaves a cancelled turn, not a pending one."""
    filename = indexed_document(client, app)

    response = client.post("/response", json={"query": "what?", "filename": filename})
    stream = response.response
    assert next(stream).startswith(b"event: token")
    stream.close()

    turn = turns_of(repositories, filename)[0]
    assert turn["status"] == "cancelled"
    assert turn["question"] == "what?"
    assert turn["answer"] is None
    assert turn["failure_reason"] == "client disconnected"
    history = client.get(f"/messages?filename={filename}").get_json()["messages"]
    assert [message["text"] for message in history] == ["what?"]


def test_an_answered_question_reloads_with_its_turn_state(client, app, repositories):
    """A stored answer replays with the same order, status, and sources."""
    filename = indexed_document(client, app)
    response = client.post("/response", json={"query": "what?", "filename": filename})
    done = parse_sse(response.get_data(as_text=True))[-1][1]

    history = client.get(f"/messages?filename={filename}").get_json()["messages"]

    assert [message["sender"] for message in history] == ["user", "bot"]
    assert history[1]["text"] == "The answer is 42."
    assert history[1]["turn_status"] == "answered"
    assert history[1]["turn_sequence"] == 1
    assert history[1]["sources"] == done["sources"]


def test_each_question_gets_its_own_turn_in_the_order_asked(client, app, repositories):
    """Successive questions become successive turns, not one merged exchange."""
    filename = indexed_document(client, app)

    for question in ("first question", "second question"):
        client.post(
            "/response", json={"query": question, "filename": filename}
        ).get_data()

    turns = turns_of(repositories, filename)
    assert [turn["sequence"] for turn in turns] == [1, 2]
    assert [turn["question"] for turn in turns] == ["first question", "second question"]
    assert [turn["status"] for turn in turns] == ["answered", "answered"]
