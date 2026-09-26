"""Ordered conversation turn tests at the repository boundary."""

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from db import Conversation, Source, Turn
from repositories import ConversationRepository, FileRepository
from tests.chat_app import session_factory


@pytest.fixture
def conversation(session_factory):
    """Create one stored document and the conversation bound to it."""
    files = FileRepository(session_factory)
    file_id = files.create_file("doc.pdf", title="Doc")["id"]
    repository = ConversationRepository(session_factory)
    return repository, repository.ensure_conversation(file_id), file_id


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
