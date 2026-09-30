"""
Starting a Research Brief: what it may read, and what it is allowed to spend.

A brief is the first request here where the model decides what gets read, which
is what makes it worth having for a question across two papers and exactly what
makes it able to spend without being asked. These tests pin both ends of that:
that the pair of Documents is chosen before the first model turn and cannot be
widened afterwards, and that every limit is checked while the loop runs rather
than after it.

They are driven over HTTP because that is where the decisions are made. A
refusal that never opens a stream is an outcome a reader has to be able to tell
from a run that failed, and the only place that distinction is real is the
response itself.
"""

import json

from unittest.mock import patch

import pytest

from services.accounts.chat_settings_service import model_for
from services.answering import Refusal
from services.brief.prompts import BRIEF_PROMPT_VERSION
from services.brief.service import BriefService
from services.llm.base import ProviderTimeoutError
from repositories import build_repositories
from tests.chat_app import _Embeddings, _Vectors, in_memory_session_factory
from tests.conftest import TEST_SETTINGS
from tests.brief_app import (
    BriefTurn,
    app,
    brief_turn,
    call_tool,
    client,
    fake_brief,
    indexed_document,
    read_tool,
    repositories,
    search_tool,
    session_factory,
    two_documents,
)
from tests.sse import parse_sse
from tests.telemetry_harness import recorded_traces, recording_tracer, tracer

__all__ = [
    "app",
    "client",
    "fake_brief",
    "indexed_document",
    "recorded_traces",
    "recording_tracer",
    "repositories",
    "session_factory",
    "tracer",
    "two_documents",
]


def start(client, documents, question="how do the two disagree?"):
    """Start a brief over these Documents and return the raw response."""
    return client.post(
        "/research", json={"question": question, "documents": list(documents)}
    )


def brief(client, documents, question="how do the two disagree?"):
    """Start a brief and return its parsed event stream."""
    response = start(client, documents, question)
    assert response.mimetype == "text/event-stream", response.get_data(as_text=True)
    return parse_sse(response.get_data(as_text=True))


def last(stream):
    """Return the terminal event of a brief."""
    return stream[-1]


def tool_spans(traces):
    """Return the tool spans of the most recent brief trace, in order."""
    return [span.attributes for span in traces.last().find("tool")]


def refusals(traces):
    """Return the reason of every refused tool call in the most recent brief."""
    return [
        attributes["refusal_reason"]
        for attributes in tool_spans(traces)
        if attributes.get("refusal_reason")
    ]


def _three_searches() -> BriefTurn:
    """Return a turn that asks for three searches in one go."""
    return BriefTurn(
        calls=[
            (f"c{position}", "search_passages", {"label": "A", "query": f"q{position}"})
            for position in (1, 2, 3)
        ],
        finish_reason="tool_calls",
    )


def tool_result(factory, tool):
    """
    Return the payload one tool returned, read from the conversation the model saw.

    The scripted provider is handed the whole conversation on each turn, so the
    results a tool produced are in what the next turn was given. Reading them
    there rather than off a spy is what makes the assertion about what the model
    actually received.
    """
    for _roles, _tools, _instruction, prompt in reversed(factory.turns):
        for line in prompt.splitlines():
            if not line.startswith('{"results"') and not line.startswith('{"passages"'):
                continue
            payload = json.loads(line)
            if "results" in payload and tool == "search_passages":
                return payload["results"][0]
            if "passages" in payload and tool == "read_passages":
                return payload["passages"][0]
    raise AssertionError(f"no {tool} result reached the model")


def searches_of(app):
    """Return the Documents the vector stand-in was asked to search."""
    return [query["filename"] for query in app.config["TEST_VECTORS"].queries]


class TestSelection:
    """Which Documents a brief is allowed to read."""

    def test_a_brief_over_two_documents_runs(self, client, app, fake_brief):
        """Two indexed Documents are what a brief is for."""
        documents = two_documents(client, app)
        fake_brief.brief(search_tool("c1", "A"), brief_turn(text="A agrees."))

        assert start(client, documents).status_code == 200

    def test_the_scope_is_shown_before_the_brief_starts(self, client, app, fake_brief):
        """A reader sees which two Documents a brief reads, and under what labels."""
        documents = two_documents(client, app)
        fake_brief.brief(brief_turn(text="They agree."))

        opened = brief(client, documents)[0]

        assert opened[0] == "start"
        assert [item["label"] for item in opened[1]["documents"]] == ["A", "B"]
        assert all(item["document_id"] for item in opened[1]["documents"])

    def test_the_scope_names_the_documents_by_title(self, client, app, fake_brief):
        """The model is told which paper is which, without being told a filename."""
        documents = two_documents(client, app)
        fake_brief.brief(search_tool("c1", "A"), brief_turn(text="A agrees."))

        brief(client, documents)

        _, roles, _, prompt = fake_brief.turns[0]
        assert "[A] " in prompt
        assert documents[0] not in prompt

    def test_one_document_is_refused(self, client, app):
        """A brief is a cross-Document question, so one Document is not a scope."""
        filename = indexed_document(client, app)

        response = start(client, [filename])

        assert response.status_code == 400
        assert response.get_json()["category"] == "brief_scope_invalid"

    def test_three_documents_are_refused(self, client, app):
        """A third Document would widen the reach, so the pair is the ceiling."""
        documents = two_documents(client, app, "a.pdf", "b.pdf")
        third = indexed_document(client, app, "c.pdf")

        response = start(client, [*documents, third])

        assert response.status_code == 400
        assert response.get_json()["category"] == "brief_scope_invalid"

    def test_the_same_document_named_twice_is_refused(self, client, app):
        """One Document twice is not a pair, and would rank it against itself."""
        filename = indexed_document(client, app)

        response = start(client, [filename, filename])

        assert response.status_code == 400
        assert response.get_json()["category"] == "brief_scope_invalid"

    def test_an_unknown_document_is_refused(self, client, app):
        """A Document that does not exist cannot be part of a scope."""
        filename = indexed_document(client, app)

        response = start(client, [filename, "never-uploaded.pdf"])

        assert response.status_code == 404
        assert response.get_json()["category"] == "file_not_found"

    def test_a_stale_document_is_refused_before_any_model_call(
        self, client, app, fake_brief, repositories
    ):
        """A stale index would answer from an incompatible one, so the pair stops."""
        documents = two_documents(client, app)
        make_stale(repositories, documents[0])

        response = start(client, documents)

        assert response.status_code == 409
        assert response.get_json()["category"] == "index_stale"
        assert fake_brief.turns == []

    def test_a_document_being_reindexed_is_refused(self, client, app, repositories):
        """A Document whose vectors are being rewritten cannot be read."""
        documents = two_documents(client, app)
        assert client.post(f"/files/{documents[0]}/reindex").status_code == 201

        response = start(client, documents)

        assert response.status_code == 409
        assert response.get_json()["category"] == "document_indexing"

    def test_an_unindexed_document_is_refused(self, client, app, repositories):
        """A Document with no index has nothing for a brief to search."""
        documents = two_documents(client, app)
        repositories.files.set_processed(documents[0], False)

        response = start(client, documents)

        assert response.status_code == 409
        assert response.get_json()["category"] == "index_pending"

    def test_a_traversing_document_name_is_refused(self, client, app):
        """A filename that walks out of storage never becomes a scope."""
        response = client.post(
            "/research",
            json={"question": "q", "documents": ["../secrets.pdf", "other.pdf"]},
        )

        assert response.status_code == 400


def make_stale(repositories, filename) -> None:
    """Leave one Document's index describing a parser the app no longer serves."""
    repositories.files._update(
        filename, index_manifest=json.dumps({"parser": "a-parser-from-last-year"})
    )


class TestCapabilityGate:
    """Which models a brief may run on."""

    def test_a_model_without_tool_use_cannot_run_a_brief(
        self, client, app, fake_brief, without_capability
    ):
        """A loop that cannot be asked for tools cannot run at all."""
        documents = two_documents(client, app)
        without_capability("tool_use")

        response = start(client, documents)

        assert response.status_code == 409
        assert response.get_json()["missing_capabilities"] == ["tool_use"]
        assert fake_brief.turns == []

    def test_a_model_without_structured_output_cannot_run_a_brief(
        self, client, app, fake_brief, without_capability
    ):
        """A brief's result cannot be read back as a brief without one."""
        documents = two_documents(client, app)
        without_capability("structured_output")

        response = start(client, documents)

        assert response.status_code == 409
        assert response.get_json()["category"] == "research_brief_unsupported_model"
        assert response.get_json()["missing_capabilities"] == ["structured_output"]
        assert fake_brief.turns == []

    def test_a_blocked_brief_is_recorded_without_being_paid_for(
        self, client, app, recorded_traces, without_capability
    ):
        """A brief refused on capability is still accountable, and cost nothing."""
        documents = two_documents(client, app)
        without_capability("tool_use")

        start(client, documents)

        assert recorded_traces.traces[-1].attributes["outcome"] == "refused"
        assert recorded_traces.traces[-1].find("scope") == []


@pytest.fixture
def without_capability():
    """Take one capability away from the catalog model a brief would run on."""
    from dataclasses import replace

    from services.accounts import chat_settings_service as catalog

    def _strip(capability: str) -> None:
        catalog.MODEL_CATALOG = tuple(
            replace(entry, **{capability: False})
            if entry.id == "openai/gpt-oss-120b"
            else entry
            for entry in catalog.MODEL_CATALOG
        )

    yield _strip
    catalog.MODEL_CATALOG = tuple(
        replace(entry, tool_use=True, structured_output=True)
        for entry in catalog.MODEL_CATALOG
    )


class TestToolReach:
    """What the model may call, and what it may not."""

    def test_the_model_is_offered_exactly_the_two_read_tools(
        self, client, app, fake_brief
    ):
        """The whole tool surface is two retrieval tools and nothing else."""
        documents = two_documents(client, app)
        fake_brief.brief(search_tool("c1", "A"), brief_turn(text="They agree."))

        brief(client, documents)

        _, tools, instruction, _ = fake_brief.turns[0]
        assert tools == ["search_passages", "read_passages"]
        assert "no tool to browse the web" in instruction
        assert "delete" in instruction

    def test_a_search_reads_the_document_the_model_named(self, client, app, fake_brief):
        """A search over label B reaches B, not whichever came first."""
        documents = two_documents(client, app)
        fake_brief.brief(search_tool("c1", "B"), brief_turn(text="B says so."))

        brief(client, documents)

        assert searches_of(app) == [documents[1]]

    def test_a_label_outside_the_scope_is_refused(
        self, client, app, fake_brief, recorded_traces
    ):
        """A label the brief was not given names no Document, and is not looked up."""
        documents = two_documents(client, app)
        fake_brief.brief(
            call_tool("c1", "search_passages", {"label": "C", "query": "retention"}),
            brief_turn(text="I cannot read C."),
        )

        brief(client, documents)

        assert refusals(recorded_traces) == ["out_of_scope"]
        assert searches_of(app) == []

    def test_an_unknown_tool_is_refused(self, client, app, fake_brief, recorded_traces):
        """There is no tool to delete a Document, so asking for one is refused."""
        documents = two_documents(client, app)
        fake_brief.brief(
            call_tool("c1", "delete_document", {"filename": documents[0]}),
            brief_turn(text="I cannot delete anything."),
        )

        brief(client, documents)

        assert refusals(recorded_traces) == ["unknown_tool"]

    def test_a_tool_that_would_reach_the_workspace_is_refused(
        self, client, app, fake_brief, recorded_traces
    ):
        """Settings, uploads, code, and the web are all outside a brief's reach."""
        documents = two_documents(client, app)
        fake_brief.brief(
            call_tool("c1", "update_settings", {"api_key": "sk-x"}),
            call_tool("c2", "run_code", {"code": "print(1)"}),
            call_tool("c3", "browse", {"url": "https://example.com"}),
            call_tool("c4", "upload_file", {"path": "/etc/passwd"}),
            brief_turn(text="I have no such tools."),
        )

        brief(client, documents)

        assert refusals(recorded_traces) == ["unknown_tool"] * 4

    def test_arguments_that_do_not_match_the_schema_are_refused(
        self, client, app, fake_brief, recorded_traces
    ):
        """A search with no label would read whichever Document came first."""
        documents = two_documents(client, app)
        fake_brief.brief(
            call_tool("c1", "search_passages", {"query": "retention"}),
            brief_turn(text="I need a label."),
        )

        brief(client, documents)

        assert refusals(recorded_traces) == ["invalid_arguments"]

    def test_an_unexpected_argument_is_refused(
        self, client, app, fake_brief, recorded_traces
    ):
        """An argument the tool does not declare is one it would have to ignore."""
        documents = two_documents(client, app)
        fake_brief.brief(
            call_tool(
                "c1",
                "search_passages",
                {"label": "A", "query": "retention", "index_generation": 99},
            ),
            brief_turn(text="A says so."),
        )

        brief(client, documents)

        assert refusals(recorded_traces) == ["invalid_arguments"]

    def test_a_refused_call_never_reaches_retrieval(self, client, app, fake_brief):
        """A refused tool spends nothing, because it never searches."""
        documents = two_documents(client, app)
        fake_brief.brief(
            call_tool("c1", "search_passages", {"label": "C", "query": "retention"}),
            brief_turn(text="Out of scope."),
        )

        brief(client, documents)

        assert searches_of(app) == []

    def test_a_refused_call_is_told_what_the_brief_may_read(
        self, client, app, fake_brief
    ):
        """The model is given a reason and a way forward, not a dead end."""
        documents = two_documents(client, app)
        fake_brief.brief(
            call_tool("c1", "search_passages", {"label": "C", "query": "retention"}),
            brief_turn(text="Out of scope."),
        )

        brief(client, documents)

        roles, _, _, prompt = fake_brief.turns[1]
        assert roles[-1] == "tool"
        assert "not part of this brief" in prompt


class TestEvidenceIds:
    """The ids a brief's evidence is held and cited under."""

    def test_every_passage_a_search_returns_gets_an_id(self, client, app, fake_brief):
        """A result the model cannot cite by id is a result it will not cite."""
        documents = two_documents(client, app)
        fake_brief.brief(search_tool("c1", "A"), brief_turn(text="A says so."))

        evidence = last(brief(client, documents))[1]["evidence"]

        assert [item["evidence_id"] for item in evidence] == ["E1"]
        assert evidence[0]["rank"] == 1
        assert evidence[0]["page"] == 1

    def test_an_id_is_stable_across_the_whole_run(self, client, app, fake_brief):
        """A citation written in the first minute still points at its Passage."""
        documents = two_documents(client, app)
        fake_brief.brief(
            search_tool("c1", "A"),
            read_tool("c2", "E1"),
            brief_turn(text="A says so."),
        )

        evidence = last(brief(client, documents))[1]["evidence"]

        assert [item["evidence_id"] for item in evidence] == ["E1"]

    def test_a_passage_found_twice_is_one_piece_of_evidence(
        self, client, app, fake_brief
    ):
        """Two searches returning the same Passage do not renumber it."""
        documents = two_documents(client, app)
        fake_brief.brief(
            search_tool("c1", "A", "retention"),
            search_tool("c2", "A", "policy"),
            brief_turn(text="A says the same thing either way."),
        )

        evidence = last(brief(client, documents))[1]["evidence"]

        assert [item["evidence_id"] for item in evidence] == ["E1"]

    def test_two_documents_collect_evidence_under_their_own_labels(
        self, client, app, fake_brief
    ):
        """A cross-Document brief is about what two papers say differently."""
        documents = two_documents(client, app)
        fake_brief.brief(
            search_tool("c1", "A"),
            search_tool("c2", "B"),
            brief_turn(text="They differ."),
        )

        evidence = last(brief(client, documents))[1]["evidence"]

        assert sorted({item["label"] for item in evidence}) == ["A", "B"]

    def test_reading_an_id_this_brief_never_collected_is_refused(
        self, client, app, fake_brief, recorded_traces
    ):
        """An id from outside this run names nothing here, so nothing comes back."""
        documents = two_documents(client, app)
        fake_brief.brief(
            read_tool("c1", "E9"), brief_turn(text="I have no such evidence.")
        )

        brief(client, documents)

        assert refusals(recorded_traces) == ["unknown_evidence"]


class TestBudget:
    """What the loop is allowed to spend."""

    def test_a_brief_that_answers_stops_at_its_answer(self, client, app, fake_brief):
        """A finished brief is not run again to see whether it would say more."""
        documents = two_documents(client, app)
        fake_brief.brief(search_tool("c1", "A"), brief_turn(text="A says so."))

        done = last(brief(client, documents))[1]

        assert done["status"] == "complete"
        assert done["budget"]["turns"] == 2

    def test_a_brief_that_never_searched_is_not_grounded_in_anything(
        self, client, app, fake_brief
    ):
        """Prose with no evidence behind it is not a brief, whatever the model says."""
        documents = two_documents(client, app)
        fake_brief.brief(brief_turn(text="They agree, I am certain."))

        stream = brief(client, documents)

        assert stream[-1][0] == "abstained"

    def test_the_terminal_event_reports_the_prompt_version_and_budget(
        self, client, app, fake_brief
    ):
        """Two briefs are only comparable if both were asked, and spent, the same."""
        documents = two_documents(client, app)
        fake_brief.brief(search_tool("c1", "A"), brief_turn(text="A says so."))

        done = last(brief(client, documents))[1]

        assert done["prompt_version"] == BRIEF_PROMPT_VERSION
        assert done["budget"]["max_turns"] >= done["budget"]["turns"]
        assert done["budget"]["tool_calls"] == 1
        assert done["budget"]["tokens"] > 0

    def test_a_brief_that_runs_out_of_turns_is_marked_incomplete(
        self, client, app, fake_brief, one_turn
    ):
        """A model that never answers is stopped and labelled, not shown as done."""
        documents = two_documents(client, app)
        fake_brief.brief(search_tool("c1", "A"))

        done = last(brief(client, documents))[1]

        assert done["status"] == "incomplete"
        assert done["stopped_by"] == "turns"
        assert done["budget"]["turns"] == 1
        assert done["evidence"]

    def test_the_turn_ceiling_comes_from_settings(
        self, client, app, fake_brief, one_turn
    ):
        """A deployment bounds a brief against its own budget."""
        documents = two_documents(client, app)
        fake_brief.brief(search_tool("c1", "A"))

        done = last(brief(client, documents))[1]

        assert done["budget"]["max_turns"] == 1
        assert done["stopped_by"] == "turns"

    def test_an_identical_call_is_refused_once_it_has_been_made_twice(
        self, client, app, fake_brief, recorded_traces
    ):
        """A model re-asking the same question has stopped exploring."""
        documents = two_documents(client, app)
        fake_brief.brief(
            search_tool("c1", "A", "retention"),
            search_tool("c2", "A", "retention"),
            search_tool("c3", "A", "retention"),
        )

        brief(client, documents)

        # The last scripted turn repeats for ever, so the refusal is not a
        # single event: the third identical call and every one after it.
        assert refusals(recorded_traces) == ["repeated_call"] * 4

    def test_two_different_searches_are_not_a_repeat(
        self, client, app, fake_brief, recorded_traces
    ):
        """A brief that searches each Document once is exploring, not looping."""
        documents = two_documents(client, app)
        fake_brief.brief(
            search_tool("c1", "A", "retention"),
            search_tool("c2", "B", "retention"),
            brief_turn(text="A keeps longer than B."),
        )

        brief(client, documents)

        assert refusals(recorded_traces) == []

    def test_a_brief_that_runs_out_of_time_is_marked_incomplete(
        self, client, app, fake_brief, no_time_left
    ):
        """A brief that overran its wall clock stops with what it collected."""
        documents = two_documents(client, app)
        fake_brief.brief(search_tool("c1", "A"))

        done = last(brief(client, documents))[1]

        assert done["status"] == "incomplete"
        assert done["stopped_by"] == "time"
        assert done["evidence"]


@pytest.fixture
def one_turn(settings_obj, monkeypatch):
    """Bound a brief to a single turn."""
    monkeypatch.setattr(settings_obj.research, "max_turns", 1)


class _RunningClock:
    """
    A clock that jumps past the brief's ceiling at a chosen reading.

    A brief reads its clock a fixed number of times per turn: once to open its
    own trace, once to start its budget, once before each turn, and twice around
    each search. Counting to the reading where the jump happens is what lets a
    test run a brief out of time deterministically instead of waiting for a real
    minute to pass, and the reading is named where the fixture uses it so the
    choice is visible rather than magic.
    """

    #: The reads a brief makes before its first turn: the trace clock, the
    #: budget clock, and the stop check that lets that turn start. Two readings
    #: short of here ends a brief with nothing collected.
    READS_PER_SEARCH = 5
    READS_BEFORE_FIRST_TURN = 3

    def __init__(self, jump_after: int) -> None:
        """Stay at zero until this many readings, then run out of time."""
        self.jump_after = jump_after
        self.readings = 0

    def __call__(self) -> float:
        """Return the current reading, running out of time once past the jump."""
        self.readings += 1
        return 0.0 if self.readings <= self.jump_after else 1_000.0


@pytest.fixture
def no_time_left(request, app):
    """
    Run the brief against a clock that runs out partway through it.

    ``pytest.mark.parametrize("no_time_left", [reads], indirect=True)`` chooses
    where the clock jumps: the reads that fall before the first search end the
    brief with nothing collected, and the reads past them end it with one
    search already done.
    """
    from services.brief import service as brief_service

    jump_after = getattr(request, "param", _RunningClock.READS_PER_SEARCH)
    original = brief_service.CLOCK
    brief_service.CLOCK = _RunningClock(jump_after)
    yield brief_service.CLOCK
    brief_service.CLOCK = original


class TestProviderFailure:
    """What a brief whose model call fails reports."""

    def test_a_provider_failure_names_its_kind(self, client, app, fake_brief):
        """A dead provider is not an incomplete brief and not an empty answer."""
        documents = two_documents(client, app)
        fake_brief.brief(brief_turn(error=RuntimeError("provider down")))

        stream = brief(client, documents)

        assert last(stream)[0] == "provider_error"
        assert last(stream)[1]["category"] == "provider"

    def test_a_timeout_is_named_as_a_timeout(self, client, app, fake_brief):
        """A provider that ran out of time is not one that refused."""
        documents = two_documents(client, app)
        fake_brief.brief(brief_turn(error=ProviderTimeoutError("too slow")))

        stream = brief(client, documents)

        assert last(stream)[1]["category"] == "timeout"

    def test_a_failure_still_sends_what_the_brief_had_collected(
        self, client, app, fake_brief
    ):
        """A reader who saw evidence arrive keeps it rather than being shown nothing."""
        documents = two_documents(client, app)
        fake_brief.brief(
            search_tool("c1", "A"), brief_turn(error=RuntimeError("provider down"))
        )

        failed = last(brief(client, documents))[1]

        assert failed["evidence"]
        assert failed["status"] == "incomplete"

    def test_no_brief_is_stored_in_a_conversation(
        self, client, app, fake_brief, repositories
    ):
        """A brief spans two Documents, so it belongs to neither Conversation."""
        documents = two_documents(client, app)
        fake_brief.brief(search_tool("c1", "A"), brief_turn(text="A says so."))

        brief(client, documents)

        for document in documents:
            messages = client.get("/messages", query_string={"filename": document})
            assert messages.get_json()["messages"] == []


class TestCancellation:
    """A reader who stops a brief, or walks away from one."""

    def test_a_stopped_brief_returns_what_it_had_collected(self):
        """Stop is not leave: the reader gets the evidence, marked incomplete."""
        service, resolved, _ = _brief_over_two_documents()

        stopped = last(_stop_after_first_search(service, resolved))

        assert stopped.name == "done"
        assert stopped.payload["status"] == "incomplete"
        assert stopped.payload["stopped_by"] == "cancelled"
        assert stopped.payload["evidence"], "evidence collected before the stop is kept"
        assert stopped.payload["complete"] is False

    def test_a_stopped_brief_records_that_it_was_stopped(self):
        """The trace says the reader stopped it, not that it finished."""
        service, resolved, traces = _brief_over_two_documents()

        _stop_after_first_search(service, resolved)

        assert traces.last().attributes["outcome"] == "incomplete"
        assert traces.last().attributes["outcome_reason"] == "cancelled"

    def test_a_stopped_brief_never_reaches_its_own_finishing_line(self):
        """A brief cut short must not go on to answer as though it had not."""
        service, resolved, _ = _brief_over_two_documents()

        events = _stop_after_first_search(service, resolved)

        assert [event.name for event in events] == ["start", "done"]
        assert "answer" not in events[-1].payload

    def test_stopping_a_brief_that_ended_is_refused(self, client):
        """A cancel for a brief that is gone says so rather than pretending."""
        response = client.post("/research/cancel", json={"brief_id": "never-ran"})

        assert response.status_code == 409
        assert response.get_json()["category"] == "brief_not_running"

    def test_a_brief_id_is_required_to_stop_one(self, client):
        """Cancelling nothing in particular is not a request the server can act on."""
        response = client.post("/research/cancel", json={})

        assert response.status_code == 400

    def test_a_cancelled_brief_is_recorded_as_cancelled(
        self, repositories, settings_obj
    ):
        """A run that stops early must not read as one that finished."""
        trace, traces = _recording_tracer()
        service = BriefService(
            settings=settings_obj,
            repositories=repositories,
            embedding_service=_Embeddings(),
            vector_service=_Vectors(),
            tracer=trace,
        )
        resolved = _resolved_brief(service, repositories, settings_obj)

        _walk_away(service, resolved)

        assert traces.last().attributes["outcome"] == "cancelled"

    def test_a_cancelled_brief_records_what_it_had_spent(
        self, repositories, settings_obj
    ):
        """A cancelled run is still a run, so the trace says what it used."""
        trace, traces = _recording_tracer()
        service = BriefService(
            settings=settings_obj,
            repositories=repositories,
            embedding_service=_Embeddings(),
            vector_service=_Vectors(),
            tracer=trace,
        )
        resolved = _resolved_brief(service, repositories, settings_obj)

        _walk_away(service, resolved)

        budget = traces.last().first("budget").attributes
        assert budget["stopped_by"] == "cancelled"
        assert budget["max_turns"] >= 1


def _brief_over_two_documents():
    """
    Return a brief over two ready Documents, with a tracer that records it.

    Built here rather than over HTTP because stopping a brief is a thing the
    reader does while the run is in flight, and a test client reads a response
    to its end. Pressing stop is the loop's own decision, so it is tested where
    the loop is.
    """
    tracer, traces = _recording_tracer()
    repositories = build_repositories(in_memory_session_factory())
    settings = TEST_SETTINGS
    service = BriefService(
        settings=settings,
        repositories=repositories,
        embedding_service=_Embeddings(),
        vector_service=_Vectors(),
        tracer=tracer,
    )
    resolved = _resolved_brief(service, repositories, settings, tracer=tracer)
    return service, resolved, traces


def _stop_after_first_search(service, resolved):
    """Run one brief, asking it to stop as soon as it has searched once."""
    seen: list[int] = []
    original = BriefService._run_tool

    def _counting(self, brief, budget, call, turn, cancel):
        result = original(self, brief, budget, call, turn, cancel)
        seen.append(turn)
        if len(seen) == 1:
            brief.cancel.set()
        return result

    with patch.object(BriefService, "_run_tool", _counting):
        return list(service.stream(resolved))


def _recording_tracer():
    """Return a tracer and the reader the brief traces are read back through."""
    from services.telemetry.redaction import Redactor
    from services.telemetry.spans import Sink, Tracer

    from tests.telemetry_harness import RecordedTraces, _RecordingExporter

    exporter = _RecordingExporter()
    return (
        Tracer([Sink(exporter=exporter, redactor=Redactor())]),
        RecordedTraces(exporter),
    )


def _resolved_brief(
    service, repositories, settings_obj, documents=("a.pdf", "b.pdf"), tracer=None
):
    """Return a brief that reached the model over two ready Documents."""
    from services.brief.scope import resolve_scope
    from services.brief.service import BriefRequest
    from services.llm.base import LLMProvider

    class _Provider(LLMProvider):
        """A provider that searches once, then would answer."""

        name = "silent"

        def _build_client(self):
            """No SDK client is needed for a brief that is abandoned."""
            return None

        def verify(self) -> None:
            """Key verification is not exercised here."""
            raise NotImplementedError

        def _generate_response(self, query, context, prior_turns=""):
            """Answer a one-shot call, which this path never makes."""
            return ""

        def _stream_response(self, query, context, prior_turns=""):
            """Yield no fragments, which this path never asks for."""
            return iter(())

        def _complete_with_tools(self, messages, tools, system_instruction):
            """Ask for one search, so a cancelled brief had something to collect."""
            from services.llm.tools import ToolTurn, tool_call

            if getattr(self, "_answered", False):
                return ToolTurn(text="A says so.", finish_reason="stop")
            self._answered = True
            return ToolTurn(
                tool_calls=(
                    tool_call("c1", "search_passages", {"label": "A", "query": "x"}),
                ),
                finish_reason="tool_calls",
            )

    for name in documents:
        _index(repositories, name)
    scope, refusal = resolve_scope(
        repositories=repositories,
        settings=settings_obj,
        filenames=list(documents),
    )
    assert scope is not None, refusal
    provider = _Provider(api_key="k", model="m")
    if tracer is not None:
        service._tracer = tracer
    resolved = service.resolve(
        BriefRequest(
            filenames=scope.filenames,
            question="how do the two disagree?",
            provider=provider,
            model=model_for("groq", "openai/gpt-oss-120b"),
        ),
        scope,
    )
    assert not isinstance(resolved, Refusal), resolved
    return resolved


def _index(repositories, filename: str) -> None:
    """Leave one Document indexed and matching the running configuration."""
    from services.indexing.manifest import runtime_manifest

    repositories.files.create_file(filename, title=filename)
    repositories.files.set_processed(filename, True)
    repositories.files._update(
        filename,
        index_generation=1,
        index_manifest=json.dumps(runtime_manifest(TEST_SETTINGS).to_dict()),
    )


def _walk_away(service, resolved) -> None:
    """Take one event from a brief and close the stream before it finishes."""
    events = service.stream(resolved)
    next(events)
    events.close()


class TestReading:
    """What the two tools are each for."""

    def test_a_search_tells_the_model_it_has_only_seen_part_of_each_passage(
        self, client, app, fake_brief
    ):
        """A search that returned everything would make reading it pointless."""
        documents = two_documents(client, app)
        fake_brief.brief(search_tool("c1", "A"), brief_turn(text="A says so."))

        brief(client, documents)

        result = fake_brief.turns[1][3]
        assert "first part of each Passage" in result
        assert "read_passages" in result

    def test_a_search_shows_a_passage_in_part(self, client, app, fake_brief):
        """A search that returned whole Passages would cost as much as reading them."""
        documents = two_documents(client, app)
        app.config["TEST_VECTORS"].script(
            {
                "content": "a passage " * 400,
                "document": documents[0],
                "chunk_index": 0,
                "score": 0.9,
                "page": 1,
            }
        )
        fake_brief.brief(search_tool("c1", "A"), brief_turn(text="A says so."))

        brief(client, documents)
        excerpt = tool_result(fake_brief, "search_passages")["excerpt"]

        assert len(excerpt) == 600
        assert len(excerpt) < len(app.config["TEST_VECTORS"].matches[0]["content"])

    def test_reading_gives_the_model_the_whole_passage(self, client, app, fake_brief):
        """The Passage the model had in full is the one a claim can rest on."""
        documents = two_documents(client, app)
        app.config["TEST_VECTORS"].script(
            {
                "content": "a passage " * 400,
                "document": documents[0],
                "chunk_index": 0,
                "score": 0.9,
                "page": 1,
            }
        )
        fake_brief.brief(
            search_tool("c1", "A"), read_tool("c2", "E1"), brief_turn(text="A says so.")
        )

        brief(client, documents)

        assert len(tool_result(fake_brief, "read_passages")["passage"]) == len(
            app.config["TEST_VECTORS"].matches[0]["content"]
        )

    def test_evidence_the_brief_read_in_full_is_marked_read(
        self, client, app, fake_brief
    ):
        """Read and found are different claims, and the record says which."""
        documents = two_documents(client, app)
        fake_brief.brief(
            search_tool("c1", "A"), read_tool("c2", "E1"), brief_turn(text="A says so.")
        )

        evidence = last(brief(client, documents))[1]["evidence"]

        assert evidence[0]["read"] is True

    def test_an_excerpt_the_model_never_read_is_marked_as_one(
        self, client, app, fake_brief
    ):
        """Found and read are different claims, and the reader is told which."""
        documents = two_documents(client, app)
        fake_brief.brief(search_tool("c1", "A"), brief_turn(text="A says so."))

        evidence = last(brief(client, documents))[1]["evidence"]

        assert evidence[0]["read"] is False

    def test_evidence_carries_the_document_title_and_its_storage_key(
        self, client, app, fake_brief
    ):
        """A reader recognises a paper by its title; the store filters by its key."""
        documents = two_documents(client, app)
        fake_brief.brief(search_tool("c1", "A"), brief_turn(text="A says so."))

        evidence = last(brief(client, documents))[1]["evidence"][0]

        assert evidence["title"]
        assert evidence["document"] == documents[0]


class TestBudgetWithinATurn:
    """One model turn can ask for several calls at once."""

    """One model turn can ask for several calls at once."""

    def test_a_turn_cannot_spend_past_the_tool_call_ceiling(
        self, client, app, fake_brief, settings_obj, monkeypatch
    ):
        """Ten calls in one turn would bypass a ceiling meant to bound retrieval."""
        monkeypatch.setattr(settings_obj.research, "max_tool_calls", 1)
        documents = two_documents(client, app)
        fake_brief.brief(_three_searches())

        brief(client, documents)

        assert len(searches_of(app)) == 1

    def test_a_call_a_ceiling_stopped_is_still_answered(  # noqa: D401
        self, client, app, fake_brief, settings_obj, monkeypatch, recorded_traces
    ):
        """
        Every call the model made is answered, even one that never ran.

        Both providers pair a result to a call by id and reject a turn whose
        calls went unanswered, so leaving one out would fail the next request
        rather than ending the brief cleanly.
        """
        monkeypatch.setattr(settings_obj.research, "max_tool_calls", 1)
        documents = two_documents(client, app)
        fake_brief.brief(_three_searches(), brief_turn(text="A says so."))

        brief(client, documents)

        answered = [
            span.attributes.get("refusal_reason") or "ran"
            for span in recorded_traces.last().find("tool")
        ]
        assert answered == ["ran", "not_run", "not_run"]

    def test_a_refused_call_is_still_answered(
        self, client, app, fake_brief, recorded_traces
    ):
        """A call the brief refuses is still answered, so the turn stays readable."""
        documents = two_documents(client, app)
        fake_brief.brief(call_tool("c1", "delete_document", {"filename": "x"}))

        brief(client, documents)

        conversation = fake_brief.turns[-1][3]
        assert "A research brief can only call search_passages" in conversation


class TestAbstention:
    """A brief whose Documents held nothing to write from."""

    def test_a_brief_that_found_nothing_says_so_rather_than_pretending(
        self, client, app, fake_brief
    ):
        """Prose with no evidence behind it is not a brief."""
        documents = two_documents(client, app)
        app.config["TEST_VECTORS"].script()
        fake_brief.brief(search_tool("c1", "A"), brief_turn(text="They agree."))

        stream = brief(client, documents)

        assert stream[-1][0] == "abstained"
        assert stream[-1][1]["reason"] == "no_evidence"

    @pytest.mark.parametrize(
        "no_time_left",
        [_RunningClock.READS_BEFORE_FIRST_TURN - 1],
        indirect=True,
    )
    def test_a_brief_stopped_before_it_found_anything_says_so(
        self, client, app, fake_brief, no_time_left
    ):
        """Nothing was collected, so there is no partial brief to show."""
        documents = two_documents(client, app)
        fake_brief.brief(search_tool("c1", "A"))

        stream = brief(client, documents)

        assert stream[-1][0] == "abstained"
        assert stream[-1][1]["stopped_by"] == "time"

    def test_a_brief_that_ran_out_of_turns_and_found_nothing_says_so(
        self, client, app, fake_brief, one_turn
    ):
        """A limit with no evidence behind it is not a partial brief either."""
        documents = two_documents(client, app)
        app.config["TEST_VECTORS"].script()
        fake_brief.brief(search_tool("c1", "A"))

        stream = brief(client, documents)

        assert stream[-1][0] == "abstained"
        assert stream[-1][1]["stopped_by"] == "turns"
