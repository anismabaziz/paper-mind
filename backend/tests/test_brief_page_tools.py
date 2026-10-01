"""
Page reading and evidence comparison: what the brief's wider tools may do.

A brief that can only search Passages cannot check an exact Page, and one
that cannot arrange what it found cannot say where two papers agree and where
they pull apart. These tests pin the two tools that close those gaps —
read_page and compare_evidence — and the bounds around them: Document scope,
Page bounds, text limits, repeated-call limits, stable ids, and a trajectory
that shows what ran without repeating the words.

They are driven over HTTP because that is where the decisions are made: a
Page outside the pair or outside the Document is refused by the loop, not by
the caller.
"""

import json

import pytest

from services.brief.compare import compare_items
from services.brief.evidence import Evidence
from tests.brief_app import (
    app,
    brief_turn,
    call_tool,
    client,
    compare_tool,
    fake_brief,
    read_page_tool,
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
    "recorded_traces",
    "recording_tracer",
    "repositories",
    "session_factory",
    "tracer",
    "two_documents",
]

RETENTION_A = "retention improves with spaced practice sessions every week in trials"
RETENTION_B = (
    "retention does not improve with spaced practice sessions every week in trials"
)


def brief(client, documents, question="how do the two disagree?"):
    """Start a brief and return its parsed event stream."""
    response = client.post(
        "/research", json={"question": question, "documents": list(documents)}
    )
    assert response.mimetype == "text/event-stream", response.get_data(as_text=True)
    return parse_sse(response.get_data(as_text=True))


def last(stream):
    """Return the terminal event of a brief."""
    return stream[-1]


def refusals(traces):
    """Return the reason of every refused tool call in the most recent brief."""
    return [
        span.attributes["refusal_reason"]
        for span in traces.last().find("tool")
        if span.attributes.get("refusal_reason")
    ]


def tool_payload(factory, key):
    """
    Return the payload one tool returned, read from the conversation.

    The scripted provider is handed the whole conversation on each turn, so
    the results a tool produced are in what the next turn was given. Reading
    them there is what makes the assertion about what the model received.
    """
    for _roles, _tools, _instruction, prompt in reversed(factory.turns):
        for line in prompt.splitlines():
            stripped = line.strip()
            if not stripped.startswith("{"):
                continue
            try:
                payload = json.loads(stripped)
            except ValueError:
                continue
            if key in payload:
                return payload
    raise AssertionError(f"no {key} result reached the model")


def script_passages(app, first, second):
    """Script two retrieval matches with distinct contents and Pages."""
    app.config["TEST_VECTORS"].script(
        {
            "content": first,
            "document": "doc.pdf",
            "chunk_index": 0,
            "score": 0.9,
            "page": 2,
        },
        {
            "content": second,
            "document": "other.pdf",
            "chunk_index": 1,
            "score": 0.8,
            "page": 9,
        },
    )


def script_pages(app, documents, first, second):
    """Script distinct Pages per Document for Page reads."""
    app.config["TEST_PARSER"].script_document_pages(
        {
            documents[0]: [first, "back matter of the first paper"],
            documents[1]: [second, "back matter of the second paper"],
        }
    )


def evidence(text, evidence_id="E1", label="A"):
    """Return one held evidence for the comparison unit tests."""
    return Evidence(
        evidence_id=evidence_id,
        label=label,
        document_id=f"doc-{label}",
        title=f"Paper {label}",
        document=f"{label}.pdf",
        chunk_index=0,
        page=2,
        rank=1,
        position=1,
        method="hybrid",
        content=text,
    )


class TestPageReading:
    """What read_page returns, and what it refuses."""

    def test_a_page_read_returns_text_as_citable_evidence(
        self, client, app, fake_brief
    ):
        """A Page read is admitted under a stable id the brief can cite."""
        documents = two_documents(client, app)
        script_pages(app, documents, RETENTION_A, RETENTION_B)
        fake_brief.brief(read_page_tool("c1", "A", 1), brief_turn(text="A says so."))

        evidence = last(brief(client, documents))[1]["evidence"]
        payload = tool_payload(fake_brief, "page")["page"]

        assert payload["evidence_id"] == "E1"
        assert payload["text"] == RETENTION_A
        assert payload["truncated"] is False
        assert evidence[0]["evidence_id"] == "E1"
        assert evidence[0]["label"] == "A"
        assert evidence[0]["page"] == 1
        assert evidence[0]["read"] is True
        assert evidence[0]["method"] == "page"
        assert evidence[0]["content"] == RETENTION_A

    def test_a_page_read_is_scoped_to_the_named_document(self, client, app, fake_brief):
        """A Page of B is B's text, not whichever Document came first."""
        documents = two_documents(client, app)
        script_pages(app, documents, RETENTION_A, RETENTION_B)
        fake_brief.brief(read_page_tool("c1", "B", 1), brief_turn(text="B says so."))

        evidence = last(brief(client, documents))[1]["evidence"]

        assert evidence[0]["document"] == documents[1]
        assert evidence[0]["label"] == "B"
        assert evidence[0]["content"] == RETENTION_B

    def test_a_page_beyond_the_document_is_refused(
        self, client, app, fake_brief, recorded_traces
    ):
        """A Page the Document does not have names nothing, and is not read."""
        documents = two_documents(client, app)
        fake_brief.brief(
            read_page_tool("c1", "A", 99),
            brief_turn(text="That Page does not exist."),
        )

        stream = brief(client, documents)

        assert refusals(recorded_traces) == ["unknown_page"]
        assert last(stream)[0] == "abstained"

    def test_a_page_read_outside_the_scope_is_refused(
        self, client, app, fake_brief, recorded_traces
    ):
        """A label the brief was not given names no Document, and is not read."""
        documents = two_documents(client, app)
        fake_brief.brief(
            read_page_tool("c1", "C", 1),
            brief_turn(text="C is not in scope."),
        )

        brief(client, documents)

        assert refusals(recorded_traces) == ["out_of_scope"]

    @pytest.mark.parametrize("page", [0, -3, 100001, "1", 1.5, True, None])
    def test_a_page_that_is_not_a_bounded_integer_is_refused(
        self, client, app, fake_brief, recorded_traces, page
    ):
        """The schema admits a Page number, not a string, a float, or an absurd integer — each is refused before any Document is opened."""
        documents = two_documents(client, app)
        fake_brief.brief(
            call_tool("c1", "read_page", {"label": "A", "page": page}),
            brief_turn(text="That is not a Page."),
        )

        brief(client, documents)

        assert refusals(recorded_traces) == ["invalid_arguments"]

    def test_a_page_call_that_names_a_file_is_refused(
        self, client, app, fake_brief, recorded_traces
    ):
        """There is no argument that names a Document outside the pair, so a call that tries one is malformed rather than honoured."""
        documents = two_documents(client, app)
        fake_brief.brief(
            call_tool(
                "c1",
                "read_page",
                {"label": "A", "page": 1, "filename": documents[1]},
            ),
            brief_turn(text="No such argument."),
        )

        brief(client, documents)

        assert refusals(recorded_traces) == ["invalid_arguments"]

    def test_a_repeated_page_read_is_limited(
        self, client, app, fake_brief, recorded_traces
    ):
        """A model stuck re-reading one Page is stopped, and the Page keeps the id it was first given rather than collecting a second one."""
        documents = two_documents(client, app)
        script_pages(app, documents, RETENTION_A, RETENTION_B)
        same = read_page_tool("c1", "A", 1)
        fake_brief.brief(
            same,
            read_page_tool("c2", "A", 1),
            read_page_tool("c3", "A", 1),
            brief_turn(text="A says so."),
        )

        evidence = last(brief(client, documents))[1]["evidence"]

        assert refusals(recorded_traces) == ["repeated_call"]
        assert [item["evidence_id"] for item in evidence] == ["E1"]

    def test_a_long_page_is_truncated(self, client, app, fake_brief):
        """A Page longer than the bound arrives in part, and says so."""
        documents = two_documents(client, app)
        long_page = "retention " * 700
        app.config["TEST_PARSER"].script_pages(long_page)
        fake_brief.brief(read_page_tool("c1", "A", 1), brief_turn(text="A says so."))

        brief(client, documents)
        result = tool_payload(fake_brief, "page")
        payload = result["page"]

        assert payload["truncated"] is True
        assert len(payload["text"]) == 4000
        assert payload["total_chars"] == len(long_page)
        assert "truncated" in result["note"]

    def test_a_blank_page_collects_no_evidence(self, client, app, fake_brief):
        """A Page with no text is an honest empty result, not a citation."""
        documents = two_documents(client, app)
        app.config["TEST_PARSER"].script_pages("", "back matter")
        fake_brief.brief(
            read_page_tool("c1", "A", 1),
            brief_turn(text="That Page is blank."),
        )

        stream = brief(client, documents)
        payload = tool_payload(fake_brief, "page")["page"]

        assert payload["evidence_id"] is None
        assert payload["text"] == ""
        assert last(stream)[0] == "abstained"

    def test_a_reindexed_document_refuses_the_page(
        self, client, app, fake_brief, repositories, recorded_traces, monkeypatch
    ):
        """A Document reindexed since the brief started is no longer the Document the model was shown, so its Pages are out of scope."""
        from tests.brief_app import BriefProvider

        documents = two_documents(client, app)
        real_get_file = repositories.files.get_file
        state = {"reindexed": False}

        def get_file(filename):
            record = real_get_file(filename)
            if state["reindexed"] and record is not None and filename == documents[0]:
                return {**record, "index_generation": 9999}
            return record

        monkeypatch.setattr(repositories.files, "get_file", get_file)
        real_turn = BriefProvider._complete_with_tools

        def first_turn(self, messages, tools, system_instruction):
            state["reindexed"] = True
            return real_turn(self, messages, tools, system_instruction)

        monkeypatch.setattr(BriefProvider, "_complete_with_tools", first_turn)
        fake_brief.brief(
            read_page_tool("c1", "A", 1),
            brief_turn(text="That Document changed."),
        )

        stream = brief(client, documents)

        assert refusals(recorded_traces) == ["out_of_scope"]
        assert last(stream)[0] == "abstained"

    def test_page_text_can_be_cited_in_the_final_brief(self, client, app, fake_brief):
        """A Page read joins the evidence a claim may rest on, through retrieval or through comparison alike."""
        documents = two_documents(client, app)
        script_pages(app, documents, RETENTION_A, RETENTION_B)
        script_passages(app, RETENTION_A, RETENTION_B)
        fake_brief.brief(
            search_tool("c1", "A"),
            read_page_tool("c2", "B", 1),
            compare_tool("c3", "E1", "E3"),
            brief_turn(
                text=(
                    "Both papers speak to retention. "
                    "<brief>"
                    '{"summary": "Both papers speak to retention.", '
                    '"claims": [{"claim": "Spaced practice helps retention", '
                    '"supports": ["E1", "E3"]}], '
                    '"gaps": [], "abstained": false}'
                    "</brief>"
                )
            ),
        )

        terminal = last(brief(client, documents))[1]

        assert terminal["invalid_citations"] == []
        assert terminal["claims"][0]["status"] == "supported"
        assert terminal["claims"][0]["supports"] == ["E1", "E3"]


class TestEvidenceComparison:
    """What compare_evidence arranges, and what it never collects."""

    def test_comparison_collects_nothing_new(self, client, app, fake_brief):
        """Arranging evidence is not retrieving: the ledger holds what the searches found, and the comparison links it rather than adding to it."""
        documents = two_documents(client, app)
        script_passages(app, RETENTION_A, RETENTION_B)
        fake_brief.brief(
            search_tool("c1", "A"),
            compare_tool("c2", "E1", "E2"),
            brief_turn(text="They agree."),
        )

        terminal = last(brief(client, documents))[1]
        payload = tool_payload(fake_brief, "compared")

        assert [item["evidence_id"] for item in terminal["evidence"]] == [
            "E1",
            "E2",
        ]
        assert [item["evidence_id"] for item in payload["compared"]] == [
            "E1",
            "E2",
        ]
        assert set(payload) >= {
            "compared",
            "similarities",
            "differences",
            "contradictions",
        }

    def test_every_comparison_entry_links_held_evidence(self, client, app, fake_brief):
        """No entry may introduce an id the tools never returned: the whole comparison is traceable to what the brief holds."""
        documents = two_documents(client, app)
        script_passages(app, RETENTION_A, RETENTION_B)
        fake_brief.brief(
            search_tool("c1", "A"),
            compare_tool("c2", "E1", "E2"),
            brief_turn(text="They agree."),
        )

        brief(client, documents)
        payload = tool_payload(fake_brief, "compared")
        linked = set()
        for section in (
            payload["similarities"],
            payload["differences"],
            payload["contradictions"],
        ):
            for entry in section:
                linked.update(entry["evidence_ids"])

        assert linked <= {"E1", "E2"}
        assert payload["compared"][0]["label"] == "A"

    def test_agreeing_evidence_is_named_similar_not_contradicting(
        self, client, app, fake_brief
    ):
        """Two Passages about the same finding share their topic terms, and with no contrast cue between them nothing is flagged against them."""
        agreed = "retention improves with spaced practice sessions every week"
        documents = two_documents(client, app)
        script_passages(app, agreed, agreed + " in trials")
        fake_brief.brief(
            search_tool("c1", "A"),
            compare_tool("c2", "E1", "E2"),
            brief_turn(text="They agree."),
        )

        brief(client, documents)
        payload = tool_payload(fake_brief, "compared")

        assert payload["similarities"]
        assert payload["similarities"][0]["evidence_ids"] == ["E1", "E2"]
        assert payload["contradictions"] == []

    def test_conflicting_evidence_is_flagged_as_a_candidate(
        self, client, app, fake_brief
    ):
        """Two Passages sharing a topic while one side pushes back are flagged with the reason — a prompt to judge, not a verdict."""
        documents = two_documents(client, app)
        script_passages(app, RETENTION_A, RETENTION_B)
        fake_brief.brief(
            search_tool("c1", "A"),
            search_tool("c2", "B"),
            compare_tool("c3", "E1", "E4"),
            brief_turn(text="They disagree."),
        )

        brief(client, documents)
        payload = tool_payload(fake_brief, "compared")

        assert payload["similarities"]
        assert len(payload["contradictions"]) == 1
        candidate = payload["contradictions"][0]
        assert candidate["evidence_ids"] == ["E1", "E4"]
        assert candidate["shared_terms"]

    def test_compared_evidence_is_marked_read(self, client, app, fake_brief):
        """A comparison shows the evidence in full, so what it showed is recorded as read rather than as merely found."""
        documents = two_documents(client, app)
        script_passages(app, RETENTION_A, RETENTION_B)
        fake_brief.brief(
            search_tool("c1", "A"),
            compare_tool("c2", "E1", "E2"),
            brief_turn(text="They agree."),
        )

        evidence = last(brief(client, documents))[1]["evidence"]

        assert [item["read"] for item in evidence] == [True, True]

    def test_comparing_unknown_evidence_is_refused(
        self, client, app, fake_brief, recorded_traces
    ):
        """An id the brief never collected names nothing to compare."""
        documents = two_documents(client, app)
        script_passages(app, RETENTION_A, RETENTION_B)
        fake_brief.brief(
            search_tool("c1", "A"),
            compare_tool("c2", "E1", "E99"),
            brief_turn(text="No such evidence."),
        )

        brief(client, documents)

        assert refusals(recorded_traces) == ["unknown_evidence"]

    def test_comparing_an_evidence_with_itself_is_refused(
        self, client, app, fake_brief, recorded_traces
    ):
        """A comparison of one evidence named twice is not a comparison."""
        documents = two_documents(client, app)
        script_passages(app, RETENTION_A, RETENTION_B)
        fake_brief.brief(
            search_tool("c1", "A"),
            compare_tool("c2", "E1", "E1"),
            brief_turn(text="Nothing to compare."),
        )

        brief(client, documents)

        assert refusals(recorded_traces) == ["invalid_arguments"]

    def test_an_oversized_label_is_refused(
        self, client, app, fake_brief, recorded_traces
    ):
        """A label longer than the schema allows is malformed, not a scope."""
        documents = two_documents(client, app)
        fake_brief.brief(
            read_page_tool("c1", "ABCDEFGHI", 1),
            brief_turn(text="No such label."),
        )

        brief(client, documents)

        assert refusals(recorded_traces) == ["invalid_arguments"]

    def test_comparing_too_many_at_once_is_refused(
        self, client, app, fake_brief, recorded_traces
    ):
        """Eleven ids exceed the bound, whatever they name."""
        documents = two_documents(client, app)
        fake_brief.brief(
            compare_tool("c1", *[f"E{position}" for position in range(1, 12)]),
            brief_turn(text="Too many."),
        )

        brief(client, documents)

        assert refusals(recorded_traces) == ["invalid_arguments"]

    @pytest.mark.parametrize("bad_id", ["e1", "X1", 5, "", "E"])
    def test_comparison_ids_must_be_evidence_ids(
        self, client, app, fake_brief, recorded_traces, bad_id
    ):
        """Anything that is not an evidence id is malformed, not unknown."""
        documents = two_documents(client, app)
        script_passages(app, RETENTION_A, RETENTION_B)
        fake_brief.brief(
            search_tool("c1", "A"),
            call_tool("c2", "compare_evidence", {"evidence_ids": ["E1", bad_id]}),
            brief_turn(text="Malformed."),
        )

        brief(client, documents)

        assert refusals(recorded_traces) == ["invalid_arguments"]


class TestPageAndComparisonTrace:
    """What the redacted trajectory shows about the wider tools."""

    def test_tool_calls_record_provenance_latency_and_budget(
        self, client, app, fake_brief, recorded_traces
    ):
        """Every wider call lands on the trace with its provenance and its cost — and none of the Document's words lands there with it."""
        documents = two_documents(client, app)
        script_pages(app, documents, RETENTION_A, RETENTION_B)
        script_passages(app, RETENTION_A, RETENTION_B)
        fake_brief.brief(
            search_tool("c1", "A"),
            read_page_tool("c2", "B", 1),
            compare_tool("c3", "E1", "E2"),
            brief_turn(text="Both papers speak to retention."),
        )

        brief(client, documents)
        trace = recorded_traces.last()
        tools = [span.attributes for span in trace.find("tool")]
        budget = trace.first("budget").attributes

        assert [span["tool"] for span in tools] == [
            "search_passages",
            "read_page",
            "compare_evidence",
        ]
        page_span = tools[1]
        assert page_span["evidence_ids"] == ["E3"]
        assert page_span["page"] == 1
        assert page_span["latency_ms"] >= 0
        assert tools[2]["evidence_ids"] == ["E1", "E2"]
        assert [entry["tool"] for entry in budget["tool_trajectory"]] == [
            "search_passages",
            "read_page",
            "compare_evidence",
        ]
        assert budget["tool_calls"] == 3
        recorded = json.dumps([tools, budget["tool_trajectory"]])
        assert RETENTION_A not in recorded
        assert RETENTION_B not in recorded


class TestCompareUnit:
    """The deterministic core, without the loop around it."""

    def test_the_same_evidence_compares_the_same_way(self):
        """Arranging is a pure function of what is held: no model, no clock, no retrieval stands between the evidence and its comparison."""
        held = [evidence(RETENTION_A), evidence(RETENTION_B, "E2", "B")]

        assert compare_items(held) == compare_items(held)

    def test_nothing_held_compares_to_nothing(self):
        """An empty table has no pairs, no provenance, and no candidates."""
        compared = compare_items([])

        assert compared["compared"] == []
        assert compared["similarities"] == []
        assert compared["differences"] == []
        assert compared["contradictions"] == []
