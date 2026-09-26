"""
The answer event contract over HTTP.

One stream, one terminal event. These tests pin what the route emits for each
way an answer can end — stored, provider failure, empty output, timeout,
persistence failure, cancellation, a document deleted mid-answer — and that a
success terminal event only ever follows a committed Turn.
"""

import pytest

from services.accounts.chat_settings_service import model_for
from services.llm.base import ProviderTimeoutError
from tests.chat_app import (
    StreamPlan,
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
__all__ = [
    "StreamPlan",
    "app",
    "client",
    "fake_chat",
    "repositories",
    "session_factory",
]

TERMINAL_EVENTS = {
    "done",
    "abstained",
    "provider_error",
    "persistence_error",
    "cancelled",
}


def ask(client, filename, query="what?", **kwargs):
    """Ask one question and return the parsed event stream."""
    response = client.post(
        "/response", json={"query": query, "filename": filename}, **kwargs
    )
    assert response.mimetype == "text/event-stream"
    return parse_sse(response.get_data(as_text=True))


def names(events):
    """Return the event names in the order they were sent."""
    return [name for name, _ in events]


def only(events, name):
    """Return the payload of the one event with this name."""
    return [payload for event, payload in events if event == name][0]


def fake_chat_budget(repositories):
    """Return the budget the route resolved for the stored provider settings."""
    stored = repositories.app_settings.get_app_settings()
    return model_for(stored["provider"], stored["model"]).chat_budget()


def test_a_stream_opens_with_the_turn_it_committed(client, app, repositories):
    """The first event names the Turn the question was recorded as."""
    filename = indexed_document(client, app)

    events = ask(client, filename)

    assert names(events)[0] == "start"
    turn = turns_of(repositories, filename)[0]
    assert only(events, "start")["turn_id"] == turn["id"]


def test_a_stored_answer_ends_the_stream_with_success(client, app, fake_chat):
    """The happy path is tokens, then a done that names the sources."""
    filename = indexed_document(client, app)
    fake_chat.stream("The answer ", "is 42.")

    events = ask(client, filename)

    assert names(events) == ["start", "token", "token", "done"]
    assert only(events, "done")["sources"][0]["content"] == "chunk about topic"
    assert only(events, "done")["retrieval"]["method"] == "hybrid"


def test_a_success_event_follows_the_committed_turn_and_its_sources(
    client, app, repositories
):
    """Nothing is announced as stored before the store holds it."""
    filename = indexed_document(client, app)
    seen: list[list[dict]] = []
    original = repositories.conversations.complete_turn

    def _watch(turn_id, answer, sources):
        """Record the stored turn, then let the real write happen."""
        result = original(turn_id, answer, sources)
        seen.append(turns_of(repositories, filename))
        return result

    repositories.conversations.complete_turn = _watch
    events = ask(client, filename)

    assert len(seen) == 1
    turn = seen[0][0]
    assert turn["status"] == "answered"
    assert turn["answer"] == "The answer is 42."
    assert turn["sources"][0]["document"] == "doc.pdf"
    assert names(events)[-1] == "done"


def test_a_turn_closed_before_the_answer_saves_is_not_reported_as_stored(
    client, app, repositories, monkeypatch
):
    """A late outcome that finds the Turn already ended is a save failure."""
    filename = indexed_document(client, app)
    monkeypatch.setattr(
        repositories.conversations, "complete_turn", lambda *a, **k: False
    )

    events = ask(client, filename)

    assert names(events) == ["start", "token", "token", "persistence_error"]
    assert "saved" in only(events, "persistence_error")["error"]


def test_a_provider_failure_ends_the_stream_and_the_turn(
    client, app, fake_chat, repositories
):
    """A provider that dies mid-answer is reported as a provider error."""
    filename = indexed_document(client, app)
    fake_chat.fail_with(RuntimeError("provider down"), before_output=True)

    events = ask(client, filename)

    assert names(events) == ["start", "provider_error"]
    assert only(events, "provider_error")["category"] == "provider"
    turn = turns_of(repositories, filename)[0]
    assert turn["status"] == "failed"
    assert turn["failure_reason"] == "provider failure"
    assert turn["answer"] in only(events, "provider_error")["error"]
    history = client.get(f"/messages?filename={filename}").get_json()["messages"]
    assert [message["turn_status"] for message in history] == ["failed", "failed"]


def test_a_provider_failure_after_visible_output_is_not_retried(
    client, app, fake_chat, repositories
):
    """A failure past the first fragment is reported, not spent again."""
    filename = indexed_document(client, app)
    fake_chat.script(
        StreamPlan(fragments=["Half an answer"], error=TimeoutError("upstream reset")),
        StreamPlan(fragments=["A different answer."]),
    )

    events = ask(client, filename)

    assert names(events) == ["start", "token", "provider_error"]
    assert fake_chat.attempts == 1
    turn = turns_of(repositories, filename)[0]
    assert turn["status"] == "failed"
    assert turn["failure_reason"] == "provider failure"
    assert (
        turns_of(repositories, filename)[0]["answer"]
        in only(events, "provider_error")["error"]
    )


def test_a_transient_failure_before_visible_output_is_retried(
    client, app, fake_chat, repositories
):
    """A dropped connection before the first fragment gets one more attempt."""
    filename = indexed_document(client, app)
    fake_chat.script(
        StreamPlan(
            error=ConnectionError("connection reset by peer"), before_output=True
        ),
        StreamPlan(fragments=["Recovered."]),
    )

    events = ask(client, filename)

    assert fake_chat.attempts == 2
    assert names(events) == ["start", "token", "done"]
    assert turns_of(repositories, filename)[0]["status"] == "answered"


def test_an_empty_answer_is_reported_as_its_own_outcome(
    client, app, fake_chat, repositories
):
    """No answer text is a provider outcome, not a stored blank reply."""
    filename = indexed_document(client, app)
    fake_chat.stream()

    events = ask(client, filename)

    assert names(events) == ["start", "provider_error"]
    assert only(events, "provider_error")["category"] == "empty_output"
    turn = turns_of(repositories, filename)[0]
    assert turn["status"] == "failed"
    assert turn["failure_reason"] == "empty answer"
    assert turn["answer"] == only(events, "provider_error")["error"]


def test_a_stalled_provider_is_reported_as_a_timeout(
    client, app, fake_chat, repositories
):
    """Running out of time reads differently from the model being down."""
    filename = indexed_document(client, app)
    fake_chat.fail_with(ProviderTimeoutError("exceeded 90.0s"))

    events = ask(client, filename)

    assert only(events, "provider_error")["category"] == "timeout"
    turn = turns_of(repositories, filename)[0]
    assert turn["status"] == "failed"
    assert turn["failure_reason"] == "provider timeout"


def test_an_answer_that_cannot_be_saved_is_reported_as_a_persistence_error(
    client, app, repositories, monkeypatch
):
    """A store failure reads as its own outcome, with the turn closed."""
    filename = indexed_document(client, app)
    monkeypatch.setattr(
        repositories.conversations,
        "complete_turn",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("postgres down")),
    )

    events = ask(client, filename)

    assert names(events) == ["start", "token", "token", "persistence_error"]
    turn = turns_of(repositories, filename)[0]
    assert turn["status"] == "failed"
    assert turn["failure_reason"] == "answer could not be saved"
    assert turn["answer"] in only(events, "persistence_error")["error"]


def test_a_document_deleted_mid_answer_ends_the_stream_as_cancelled(
    client, app, repositories
):
    """An answer that can no longer be stored is a cancellation, not a success."""
    filename = indexed_document(client, app)
    response = client.post("/response", json={"query": "what?", "filename": filename})
    stream = response.response
    assert next(stream).startswith(b"event: start")
    next(stream)
    assert client.delete(f"/files/remove?path={filename}").status_code == 200

    rest = b"".join(stream).decode()

    assert "event: cancelled" in rest
    assert "event: done" not in rest


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        ({"fragments": ["The answer ", "is 42."]}, "done"),
        (
            {"error": RuntimeError("down"), "before_output": True},
            "provider_error",
        ),
        ({"fragments": []}, "provider_error"),
        ({"error": ProviderTimeoutError("late")}, "provider_error"),
    ],
    ids=["stored", "provider-failure", "empty-output", "timeout"],
)
def test_every_outcome_ends_the_stream_with_exactly_one_terminal_event(
    client, app, fake_chat, script, expected
):
    """A client can always tell how the stream ended, from one event."""
    filename = indexed_document(client, app)
    fake_chat.script(StreamPlan(**script))

    events = ask(client, filename)

    terminal = [name for name in names(events) if name in TERMINAL_EVENTS]
    assert terminal == [expected]
    assert names(events)[0] == "start"


def test_an_answer_with_no_evidence_ends_the_stream_as_an_abstention(
    client, app, fake_chat
):
    """No evidence is its own terminal event, not an empty or failed answer."""
    filename = indexed_document(client, app)
    app.config["TEST_VECTORS"].script()

    events = ask(client, filename)

    assert names(events) == ["start", "abstained"]
    assert fake_chat.streamed == []


def test_a_disconnected_client_records_a_cancelled_turn(client, app, repositories):
    """Closing the stream mid-answer leaves a terminal Turn, not a pending one."""
    filename = indexed_document(client, app)

    response = client.post("/response", json={"query": "what?", "filename": filename})
    stream = response.response
    assert next(stream).startswith(b"event: start")
    assert next(stream).startswith(b"event: token")
    stream.close()

    turn = turns_of(repositories, filename)[0]
    assert turn["status"] == "cancelled"
    assert turn["question"] == "what?"
    assert turn["answer"] is None
    assert turn["failure_reason"] == "client disconnected"
    history = client.get(f"/messages?filename={filename}").get_json()["messages"]
    assert [message["text"] for message in history] == ["what?"]


def test_a_cancelled_turn_is_never_later_overwritten_by_the_answer(
    client, app, repositories
):
    """The generator that is being closed stops where the cancel happened."""
    filename = indexed_document(client, app)

    response = client.post("/response", json={"query": "what?", "filename": filename})
    stream = response.response
    next(stream)
    next(stream)
    stream.close()

    turn = turns_of(repositories, filename)[0]
    assert turn["status"] == "cancelled"
    assert turn["answer"] is None


def test_a_truncated_answer_reports_the_finish_reason_that_ended_it(
    client, app, fake_chat
):
    """An answer stopped by the token cap is stored, flagged as truncated."""
    filename = indexed_document(client, app)
    fake_chat.script(StreamPlan(fragments=["Half an answer"], finish_reason="length"))

    events = ask(client, filename)

    done = only(events, "done")
    assert done["finish_reason"] == "length"
    assert done["truncated"] is True


def test_a_complete_answer_reports_no_truncation(client, app, fake_chat):
    """A finished answer says so, so the browser shows nothing extra."""
    filename = indexed_document(client, app)
    fake_chat.script(StreamPlan(finish_reason="stop"))

    done = only(ask(client, filename), "done")

    assert done["truncated"] is False


def test_an_answer_the_app_cut_short_is_stored_but_flagged_as_unproven(
    client, app, fake_chat, repositories
):
    """A provider that ignores its token cap cannot make the answer unsaveable."""
    filename = indexed_document(client, app)
    budget = fake_chat_budget(repositories)
    # The provider keeps going past its budget, so the app stops reading and
    # the provider never gets to report a finish reason.
    fake_chat.script(
        StreamPlan(
            fragments=["x" * 4_000, "y" * 4_000, "z" * 4_000], finish_reason=None
        )
    )

    done = only(ask(client, filename), "done")

    assert len(turns_of(repositories, filename)[0]["answer"]) == budget.max_answer_chars
    assert done["finish_reason"] is None
    assert done["truncated"] is True


def test_each_question_gets_its_own_turn_in_the_order_asked(client, app, repositories):
    """Successive questions become successive Turns, not one merged exchange."""
    filename = indexed_document(client, app)

    for question in ("first question", "second question"):
        ask(client, filename, question)

    turns = turns_of(repositories, filename)
    assert [turn["sequence"] for turn in turns] == [1, 2]
    assert [turn["question"] for turn in turns] == ["first question", "second question"]
    assert [turn["status"] for turn in turns] == ["answered", "answered"]


def test_an_answered_question_reloads_with_its_turn_state(client, app, repositories):
    """A stored answer replays with the same order, status, and sources."""
    filename = indexed_document(client, app)
    done = only(ask(client, filename), "done")

    history = client.get(f"/messages?filename={filename}").get_json()["messages"]

    assert [message["sender"] for message in history] == ["user", "bot"]
    assert history[1]["text"] == "The answer is 42."
    assert history[1]["turn_status"] == "answered"
    assert history[1]["turn_sequence"] == 1
    assert history[1]["sources"] == done["sources"]
