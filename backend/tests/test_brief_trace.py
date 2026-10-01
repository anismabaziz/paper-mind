"""
What one Research Brief's trace says, and what it must not.

A brief is the first request here where the interesting record is a trajectory
rather than one call: what the model reached for, in what order, what came back,
and what the loop was allowed to spend while it did. These tests read that trace
the way an operator would — through the same tracer the application exports to —
and pin both halves of the rule: the trajectory, ranks, and budget are all on it,
and the question, the answer, the search queries, and the Passages are not.
"""

import json

import pytest

from services.brief.prompts import BRIEF_PROMPT_VERSION
from services.telemetry.brief import BUDGET, OUTCOME, SCOPE, TOOL
from tests.brief_app import (
    app,
    brief_turn,
    call_tool,
    client,
    fake_brief,
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
    "repositories",
    "session_factory",
    "recorded_traces",
    "recording_tracer",
    "tracer",
    "two_documents",
]

QUESTION = "how do the two disagree about retention?"


def brief(client, documents):
    """Start a brief and return its parsed event stream."""
    response = client.post(
        "/research", json={"question": QUESTION, "documents": list(documents)}
    )
    assert response.mimetype == "text/event-stream"
    return parse_sse(response.get_data(as_text=True))


def recorded(trace):
    """Return what the application would have exported for this trace."""
    return json.dumps(trace.to_dict())


class TestCorrelation:
    """What one brief's trace is about."""

    def test_one_brief_produces_one_trace(self, client, app, recorded_traces):
        """A brief a reader ran is a run an operator can find again."""
        documents = two_documents(client, app)

        brief(client, documents)

        assert len(recorded_traces.traces) == 1

    def test_the_trace_names_the_two_documents_it_may_read(
        self, client, app, recorded_traces, repositories
    ):
        """The pair is the whole reach of the run, so it is on the record."""
        documents = two_documents(client, app)

        brief(client, documents)
        scope = recorded_traces.last().first(SCOPE)

        assert scope.attributes["document_count"] == 2
        assert [item["label"] for item in scope.attributes["documents"]] == ["A", "B"]
        assert (
            scope.attributes["documents"][0]["document_id"]
            == repositories.files.get_file(documents[0])["id"]
        )

    def test_the_trace_names_the_model_and_the_prompt_version(
        self, client, app, recorded_traces
    ):
        """Two briefs are only comparable if both were asked the same way."""
        documents = two_documents(client, app)

        brief(client, documents)

        assert (
            recorded_traces.last().attributes["prompt_version"] == BRIEF_PROMPT_VERSION
        )
        assert recorded_traces.last().attributes["model"] == "openai/gpt-oss-120b"


class TestToolTrajectory:
    """What the model reached for, and in what order."""

    def test_each_call_is_recorded_with_its_turn_and_tool(
        self, client, app, fake_brief, recorded_traces
    ):
        """The trajectory is the record; the order is part of it."""
        documents = two_documents(client, app)
        fake_brief.brief(
            search_tool("c1", "A", "retention"),
            read_tool("c2", "E1"),
            brief_turn(text="A says so."),
        )

        brief(client, documents)
        spans = recorded_traces.last().find(TOOL)

        assert [
            (span.attributes["turn"], span.attributes["tool"]) for span in spans
        ] == [
            (1, "search_passages"),
            (2, "read_passages"),
        ]

    def test_a_search_records_the_ranks_it_returned(
        self, client, app, fake_brief, recorded_traces
    ):
        """Ranks and Pages are the record of a search; the Passages are not."""
        documents = two_documents(client, app)
        fake_brief.brief(search_tool("c1", "A"), brief_turn(text="A says so."))

        brief(client, documents)
        search = recorded_traces.last().find(TOOL)[0]

        assert search.attributes["evidence_ids"] == ["E1"]
        assert search.attributes["ranks"] == [
            {"evidence_id": "E1", "rank": 1, "page": 1}
        ]
        assert search.attributes["label"] == "A"
        assert search.attributes["retrieval_method"] == "hybrid"
        assert search.attributes["latency_ms"] >= 0

    def test_a_refused_call_is_recorded_as_refused_with_its_reason(
        self, client, app, fake_brief, recorded_traces
    ):
        """What the model reached for is on the record whether or not it was allowed."""
        documents = two_documents(client, app)
        fake_brief.brief(
            call_tool("c1", "delete_document", {"filename": documents[0]}),
            brief_turn(text="I cannot delete anything."),
        )

        brief(client, documents)
        refused = recorded_traces.last().find(TOOL)[0]

        assert refused.attributes["refused"] is True
        assert refused.attributes["refusal_reason"] == "unknown_tool"
        assert refused.error_category == "unknown_tool"

    def test_the_arguments_are_recorded_by_key_not_by_value(
        self, client, app, fake_brief, recorded_traces
    ):
        """The call's shape is on the record; the query the model wrote is not."""
        documents = two_documents(client, app)
        fake_brief.brief(
            search_tool("c1", "A", "retention"), brief_turn(text="A says.")
        )

        brief(client, documents)

        assert recorded_traces.last().find(TOOL)[0].attributes["argument_keys"] == [
            "label",
            "query",
        ]


class TestTrajectory:
    """What the model reached for, in order, on the trace itself."""

    def test_the_trajectory_is_on_the_trace_not_only_on_the_spans(
        self, client, app, fake_brief, recorded_traces
    ):
        """A collector reading only the attributes sees the order, not just the calls."""
        documents = two_documents(client, app)
        fake_brief.brief(
            search_tool("c1", "A", "retention"),
            read_tool("c2", "E1"),
            brief_turn(text="A says so."),
        )

        brief(client, documents)
        trajectory = recorded_traces.last().first(BUDGET).attributes["tool_trajectory"]

        assert [step["tool"] for step in trajectory] == [
            "search_passages",
            "read_passages",
        ]
        assert [step["turn"] for step in trajectory] == [1, 2]

    def test_a_refusal_is_on_the_trajectory_with_its_reason(
        self, client, app, fake_brief, recorded_traces
    ):
        """What the model reached for is on the record whether or not it was allowed."""
        documents = two_documents(client, app)
        fake_brief.brief(
            call_tool("c1", "browse", {"url": "https://example.com"}),
            brief_turn(text="I have no such tool."),
        )

        brief(client, documents)
        trajectory = recorded_traces.last().first(BUDGET).attributes["tool_trajectory"]

        assert trajectory[0]["reason"] == "unknown_tool"

    def test_the_evidence_the_brief_selected_is_on_the_record(
        self, client, app, fake_brief, recorded_traces
    ):
        """Which Passages survived, from which Document, at what rank."""
        documents = two_documents(client, app)
        fake_brief.brief(search_tool("c1", "A"), brief_turn(text="A says so."))

        brief(client, documents)
        selected = recorded_traces.last().first(BUDGET).attributes["evidence_selected"]

        assert selected == [
            {
                "evidence_id": "E1",
                "label": "A",
                "document_id": selected[0]["document_id"],
                "rank": 1,
                "page": 1,
                "position": 1,
                "read": False,
            }
        ]

    def test_the_trajectory_carries_no_query_the_model_wrote(
        self, client, app, fake_brief, recorded_traces
    ):
        """A trajectory is a record of what was asked for, not of what was asked."""
        documents = two_documents(client, app)
        fake_brief.brief(
            search_tool("c1", "A", "three-year sunset clause"),
            brief_turn(text="A says so."),
        )

        brief(client, documents)

        assert "sunset clause" not in json.dumps(
            recorded_traces.last().first(BUDGET).attributes["tool_trajectory"]
        )


class TestBudgetSpan:
    """What the loop spent, against what it was allowed to spend."""

    def test_the_spend_is_recorded_beside_every_ceiling(
        self, client, app, fake_brief, recorded_traces
    ):
        """A brief that stopped on a limit is legible from the record alone."""
        documents = two_documents(client, app)
        fake_brief.brief(search_tool("c1", "A"), brief_turn(text="A says so."))

        brief(client, documents)
        budget = recorded_traces.last().first(BUDGET).attributes

        assert budget["turns"] == 2
        assert budget["tool_calls"] == 1
        assert budget["max_turns"] >= 2
        assert budget["max_tool_calls"] >= 1
        assert budget["max_seconds"] > 0
        assert budget["documents"] == 2

    def test_the_limit_that_stopped_a_brief_is_recorded_with_the_spend(
        self, client, app, fake_brief, one_turn, recorded_traces
    ):
        """Six of six turns is not the same fact as six turns."""
        documents = two_documents(client, app)
        fake_brief.brief(search_tool("c1", "A"))

        brief(client, documents)

        assert recorded_traces.last().first(BUDGET).attributes["stopped_by"] == "turns"

    def test_the_outcome_is_on_the_trace_attributes(
        self, client, app, fake_brief, recorded_traces
    ):
        """A trace opens with its verdict, so a failed brief is findable."""
        documents = two_documents(client, app)
        fake_brief.brief(search_tool("c1", "A"), brief_turn(text="A says so."))

        brief(client, documents)

        assert recorded_traces.last().attributes["outcome"] == "complete"
        assert recorded_traces.last().first(OUTCOME).attributes["status"] == "complete"


class TestRedaction:
    """What a brief's trace must not contain."""

    def test_the_question_is_not_in_the_trace(
        self, client, app, fake_brief, recorded_traces
    ):
        """The one text a reader would recognise is described, not repeated."""
        documents = two_documents(client, app)
        fake_brief.brief(search_tool("c1", "A"), brief_turn(text="A says so."))

        brief(client, documents)

        assert "retention" not in recorded(recorded_traces.last())

    def test_the_search_query_the_model_wrote_is_not_in_the_trace(
        self, client, app, fake_brief, recorded_traces
    ):
        """A model's paraphrase of the question is still the reader's question."""
        documents = two_documents(client, app)
        fake_brief.brief(
            search_tool("c1", "A", "three-year sunset clause"),
            brief_turn(text="A says so."),
        )

        brief(client, documents)

        assert "sunset clause" not in recorded(recorded_traces.last())

    def test_the_answer_is_not_in_the_trace(
        self, client, app, fake_brief, recorded_traces
    ):
        """The result is what the reader has on screen; the trace is not a copy."""
        documents = two_documents(client, app)
        fake_brief.brief(
            search_tool("c1", "A"), brief_turn(text="A keeps data for 30 months.")
        )

        brief(client, documents)

        assert "30 months" not in recorded(recorded_traces.last())

    def test_the_passage_text_is_not_in_the_trace(
        self, client, app, fake_brief, recorded_traces
    ):
        """The Passage the model read is not copied into a trace file."""
        documents = two_documents(client, app)
        fake_brief.brief(search_tool("c1", "A"), brief_turn(text="A says so."))

        brief(client, documents)

        assert "chunk about topic" not in recorded(recorded_traces.last())

    def test_the_api_key_is_not_in_the_trace(
        self, client, app, fake_brief, recorded_traces
    ):
        """The key that paid for the brief is not written down beside it."""
        documents = two_documents(client, app)
        fake_brief.brief(search_tool("c1", "A"), brief_turn(text="A says so."))

        brief(client, documents)

        assert "sk-test-chat-key" not in recorded(recorded_traces.last())

    def test_the_storage_filenames_are_not_in_the_trace(
        self, client, app, fake_brief, recorded_traces
    ):
        """A trace is read by operators, not by the model that was scoped."""
        documents = two_documents(client, app)
        fake_brief.brief(search_tool("c1", "A"), brief_turn(text="A says so."))

        brief(client, documents)

        assert all(
            filename not in recorded(recorded_traces.last()) for filename in documents
        )


@pytest.fixture
def one_turn(settings_obj, monkeypatch):
    """Bound a brief to a single turn."""
    monkeypatch.setattr(settings_obj.research, "max_turns", 1)
