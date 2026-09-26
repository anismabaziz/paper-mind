"""
What a question gets when the paper has nothing to answer it with.

Retrieval that finds nothing, or finds something that cannot be cited, is not a
provider failure and not an empty answer: there is no evidence, so the app says
so itself and no paid model is called. These tests pin that distinction at the
HTTP boundary, on the Turn it leaves behind, and against the provider calls it
does not make.
"""

import pytest

from services.abstention import (
    ABSTENTION_MESSAGES,
    EVIDENCE_UNUSABLE,
    NO_EVIDENCE,
)
from services.retrieval.base import VectorStoreUnavailableError
from tests.chat_app import (
    DEFAULT_MATCH,
    app,
    client,
    fake_chat,
    indexed_document,
    repositories,
    session_factory,
    turns_of,
)
from tests.sse import parse_sse

# Fixtures come from tests.chat_app; they are re-exported so pytest finds them.
__all__ = ["app", "client", "fake_chat", "repositories", "session_factory"]


def ask(client, filename, query="what does it say?"):
    """Ask one question and return the HTTP response."""
    return client.post("/response", json={"query": query, "filename": filename})


def events(response):
    """Return the parsed event stream of a chat response."""
    return parse_sse(response.get_data(as_text=True))


def only_terminal(response):
    """Return the one terminal event of a chat response."""
    terminal = [payload for _, payload in events(response)][-1]
    return terminal


def test_an_empty_collection_abstains_without_calling_the_provider(
    client, app, fake_chat
):
    """No passages at all is answered by the app, at no cost."""
    filename = indexed_document(client, app)
    app.config["TEST_VECTORS"].script()

    response = ask(client, filename)

    assert response.status_code == 200
    names = [name for name, _ in events(response)]
    assert names == ["start", "abstained"]
    terminal = only_terminal(response)
    assert terminal == {
        "abstained": True,
        "message": terminal["message"],
        "reason": NO_EVIDENCE,
        "retrieved": 0,
        "retrieval": {
            "method": "hybrid",
            "outcome": "empty",
            "original_query": "what does it say?",
            "expanded_query": "what does it say?",
            "query_expansion": "none",
            "dropped_turns": 0,
            "dropped_sources": 0,
        },
    }
    assert terminal["message"] == ABSTENTION_MESSAGES[NO_EVIDENCE]
    assert fake_chat.streamed == []
    assert fake_chat.attempts == 0


def test_the_abstention_is_stored_as_a_completed_turn_with_no_sources(
    client, app, repositories
):
    """The exchange is on record, with the reason and nothing invented."""
    filename = indexed_document(client, app)
    app.config["TEST_VECTORS"].script()

    terminal = only_terminal(ask(client, filename))

    turn = turns_of(repositories, filename)[-1]
    assert turn["status"] == "abstained"
    assert turn["abstention_reason"] == NO_EVIDENCE
    assert turn["answer"] == terminal["message"]
    assert turn["failure_reason"] is None
    assert turn["sources"] == []
    assert turn["completed_at"] is not None


@pytest.mark.parametrize(
    "match",
    [
        pytest.param({**DEFAULT_MATCH, "content": "   "}, id="blank_content"),
        pytest.param({**DEFAULT_MATCH, "content": ""}, id="no_content"),
        pytest.param({**DEFAULT_MATCH, "document": ""}, id="no_document"),
    ],
)
def test_matched_but_uncitable_evidence_is_its_own_reason(
    client, app, fake_chat, match
):
    """A result that cannot be quoted or cited is not evidence to answer from."""
    filename = indexed_document(client, app)
    app.config["TEST_VECTORS"].script(match)

    terminal = only_terminal(ask(client, filename))

    assert terminal["reason"] == EVIDENCE_UNUSABLE
    assert terminal["retrieved"] == 1
    assert fake_chat.streamed == []


def test_one_citable_match_among_several_is_answered(client, app, fake_chat):
    """Usable evidence wins over unusable matches beside it."""
    filename = indexed_document(client, app)
    app.config["TEST_VECTORS"].script(
        {**DEFAULT_MATCH, "content": ""},
        {**DEFAULT_MATCH, "chunk_index": 1},
    )

    response = ask(client, filename)

    assert response.status_code == 200
    assert only_terminal(response)["done"] is True
    assert len(fake_chat.streamed) == 1


def test_a_stale_index_is_refused_rather_than_declared_without_evidence(
    client, app, repositories, settings_obj, monkeypatch
):
    """A stale index is a reindex away, not an abstention."""
    filename = indexed_document(client, app)
    app.config["TEST_VECTORS"].script()
    monkeypatch.setattr(settings_obj.embedding, "revision", "a-newer-revision")

    response = ask(client, filename)

    assert response.status_code == 409
    body = response.get_json()
    assert body["category"] == "index_stale"
    assert body["action"] == "reindex"
    assert turns_of(repositories, filename) == []


def test_unavailable_retrieval_is_not_reported_as_an_abstention(
    client, app, repositories
):
    """An unreachable index is an outage the user can retry, not an absence."""
    filename = indexed_document(client, app)
    app.config["TEST_VECTORS"].fail_with(
        VectorStoreUnavailableError("qdrant is unreachable")
    )

    response = ask(client, filename)

    assert response.status_code == 503
    body = response.get_json()
    assert body["category"] == "vector_store_unavailable"
    assert turns_of(repositories, filename) == []


def test_a_question_asked_again_with_evidence_is_answered(
    client, app, repositories, fake_chat
):
    """Abstention is per question: the next one with evidence gets an answer."""
    filename = indexed_document(client, app)
    vectors = app.config["TEST_VECTORS"]
    vectors.script()
    first = only_terminal(ask(client, filename, "what about the second study?"))
    vectors.script(DEFAULT_MATCH)
    second = only_terminal(ask(client, filename, "what about the second study?"))

    assert first["reason"] == NO_EVIDENCE
    assert second["done"] is True
    assert len(fake_chat.streamed) == 1
    turns = turns_of(repositories, filename)
    assert [turn["status"] for turn in turns] == ["abstained", "answered"]
    assert turns[0]["sources"] == []
    assert [source["content"] for source in turns[1]["sources"]] == [
        DEFAULT_MATCH["content"]
    ]


def test_a_client_that_leaves_an_abstention_gets_one_that_is_already_closed(
    client, app, repositories
):
    """The Turn is decided before the stream opens, so there is nothing to cancel."""
    filename = indexed_document(client, app)
    app.config["TEST_VECTORS"].script()

    stream = client.post(
        "/response", json={"query": "what?", "filename": filename}
    ).response
    assert next(stream).startswith(b"event: start")
    stream.close()

    turn = turns_of(repositories, filename)[-1]
    assert turn["status"] == "abstained"
    assert turn["abstention_reason"] == NO_EVIDENCE
    assert turn["failure_reason"] is None


def test_an_abstention_is_not_carried_into_the_next_answer(client, app, fake_chat):
    """A refusal is not an answer, so it stays out of the transcript window."""
    filename = indexed_document(client, app)
    vectors = app.config["TEST_VECTORS"]
    vectors.script()
    only_terminal(ask(client, filename, "what about the second study?"))
    vectors.script(DEFAULT_MATCH)

    only_terminal(ask(client, filename, "and the third?"))

    _, context, prior_turns = fake_chat.streamed[0]
    assert context == f"[S1] {DEFAULT_MATCH['content']}"
    assert prior_turns == ""


def test_an_abstention_replays_from_history_as_an_abstention(client, app):
    """Reloading the conversation does not dress an abstention up as an answer."""
    filename = indexed_document(client, app)
    app.config["TEST_VECTORS"].script()
    only_terminal(ask(client, filename))

    messages = client.get("/messages", query_string={"filename": filename}).get_json()[
        "messages"
    ]

    bot = [message for message in messages if message["sender"] == "bot"]
    assert len(bot) == 1
    assert bot[0]["turn_status"] == "abstained"
    assert bot[0]["turn_abstention_reason"] == NO_EVIDENCE
    assert bot[0]["sources"] == []
