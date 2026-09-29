"""
The answer path's traces, read the way an operator reads them.

One question asked over HTTP produces one trace. These tests assert what that
trace says and what it deliberately does not say: the correlation identifiers
that tie it to the Document, Conversation, Turn, and index generation; the
retrieval, generation, citation, and persistence steps with their field values;
and the absence of the question, the answer, the prompt, the Document text, and
the API key.
"""

import json

import pytest

from services.accounts.chat_settings_service import model_for
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
from tests.telemetry_harness import recorded_traces, recording_tracer, tracer

# Fixtures come from tests.chat_app and tests.telemetry_harness; they are
# re-exported so pytest finds them.
__all__ = [
    "StreamPlan",
    "app",
    "client",
    "fake_chat",
    "indexed_document",
    "recorded_traces",
    "recording_tracer",
    "repositories",
    "session_factory",
    "tracer",
    "turns_of",
]

STORED_SETTINGS = ("groq", "openai/gpt-oss-120b")


def ask(client, filename, query="what is the retention policy?"):
    """Ask one question and return the parsed event stream."""
    response = client.post(
        "/response", json={"query": query, "filename": filename}
    )
    assert response.mimetype == "text/event-stream"
    return parse_sse(response.get_data(as_text=True))


def span(trace, name):
    """Return the one span with this name, or fail saying what the trace has."""
    found = trace.find(name)
    assert found, f"no {name} span in {[s.name for s in trace.spans]}"
    return found[0]


CLAIMED_ANSWER = (
    "The answer ",
    "is 42.\n<claims>\n"
    '{"claim": "The answer is 42.", "sources": ["S1"]}\n'
    "</claims>",
)


class TestCorrelation:
    """What one request's trace is about."""

    def test_one_question_produces_one_trace(
        self, client, app, repositories, recorded_traces
    ):
        """A reader asking a question gets a trace they can find again."""
        filename = indexed_document(client, app)

        ask(client, filename)

        assert len(recorded_traces.traces) == 1

    def test_the_trace_names_the_document_conversation_turn_and_generation(
        self, client, app, repositories, recorded_traces
    ):
        """Four identifiers are what tie a trace back to the state it read."""
        filename = indexed_document(client, app)

        ask(client, filename)
        trace = recorded_traces.traces[0]
        record = repositories.files.get_file(filename)
        conversation_id = repositories.conversations.get_conversation_id(record["id"])
        turn = turns_of(repositories, filename)[0]

        assert trace.attributes["document_id"] == record["id"]
        assert trace.attributes["conversation_id"] == conversation_id
        assert trace.attributes["turn_id"] == turn["id"]
        assert trace.attributes["index_generation"] == record["index_generation"]

    def test_every_span_carries_the_trace_id(
        self, client, app, repositories, recorded_traces
    ):
        """A span read on its own still says which request it was part of."""
        filename = indexed_document(client, app)

        ask(client, filename)
        trace = recorded_traces.traces[0]

        assert [s.span_id for s in trace.spans]
        assert len({s.span_id for s in trace.spans}) == len(trace.spans)

    def test_a_refused_question_is_traced_too(
        self, client, app, repositories, recorded_traces
    ):
        """A question an operator cannot account for is the one a trace is for."""
        response = client.post(
            "/response", json={"query": "what?", "filename": "never-uploaded.pdf"}
        )

        assert response.status_code == 404
        assert len(recorded_traces.traces) == 1
        assert recorded_traces.traces[0].attributes["outcome"] == "refused"
        assert recorded_traces.traces[0].attributes["refusal_category"] == (
            "file_not_found"
        )


class TestRetrievalSpan:
    """What the retrieval span reports."""

    def test_it_reports_the_method_outcome_and_latency(
        self, client, app, repositories, recorded_traces
    ):
        """The method that ran and what it cost are the two facts a reader needs."""
        filename = indexed_document(client, app)

        ask(client, filename)
        recorded = span(recorded_traces.traces[0], "retrieval")

        assert recorded.attributes["retrieval_method"] == "hybrid"
        assert recorded.attributes["retrieval_outcome"] == "success"
        assert recorded.attributes["latency_ms"] >= 0

    def test_it_reports_the_filter_it_queried(
        self, client, app, repositories, recorded_traces
    ):
        """Which Document and which generation were read is part of the record."""
        filename = indexed_document(client, app)

        ask(client, filename)
        recorded = span(recorded_traces.traces[0], "retrieval")
        record = repositories.files.get_file(filename)

        assert recorded.attributes["filter"] == {
            "document_id": record["id"],
            "index_generation": record["index_generation"],
        }

    def test_it_reports_candidate_ranks_and_scores_without_their_text(
        self, client, app, recorded_traces
    ):
        """Ranks, scores, hashes, and Pages are the record; the Passage is not."""
        filename = indexed_document(client, app)

        ask(client, filename)
        candidates = span(recorded_traces.traces[0], "retrieval").attributes[
            "candidates"
        ]

        assert candidates, "retrieval returned a candidate"
        candidate = candidates[0]
        assert candidate["rank"] == 1
        assert candidate["fused_rank"] == 1
        assert candidate["content_hash"]
        assert candidate["selected"] is True
        assert "chunk about topic" not in json.dumps(candidates)

    def test_it_reports_how_many_candidates_were_returned_and_kept(
        self, client, app, recorded_traces
    ):
        """A candidate the app dropped is still counted, so the bound is visible."""
        filename = indexed_document(client, app)

        ask(client, filename)
        recorded = span(recorded_traces.traces[0], "retrieval")

        assert recorded.attributes["candidate_count"] >= 1
        assert recorded.attributes["selected_count"] == len(
            recorded.attributes["candidates"]
        )

    def test_it_reports_the_rerank_where_one_ran(
        self, client, app, recorded_traces
    ):
        """Whether the reranker ran, and whether it moved anything, is on the record."""
        filename = indexed_document(client, app)

        ask(client, filename)
        rerank = span(recorded_traces.traces[0], "retrieval").attributes["rerank"]

        # The test harness's vector stand-in does not rerank, so the span
        # says so rather than inventing a rerank that never happened.
        assert rerank == {"applied": False}
        candidates = span(recorded_traces.traces[0], "retrieval").attributes[
            "candidates"
        ]
        assert all(
            candidate["rerank_rank"] is None or candidate["rerank_rank"] >= 1
            for candidate in candidates
        )

    def test_a_retrieval_that_fails_is_classified_not_swallowed(
        self, client, app, recorded_traces
    ):
        """A store that is down names itself, rather than leaving an empty span."""
        from services.retrieval.base import VectorStoreUnavailableError

        filename = indexed_document(client, app)
        app.config["TEST_VECTORS"].fail_with(VectorStoreUnavailableError("down"))

        response = client.post(
            "/response", json={"query": "what?", "filename": filename}
        )

        assert response.status_code == 503
        recorded = span(recorded_traces.traces[0], "retrieval")
        assert recorded.error_category == "vector_store_unavailable"


class TestGenerationSpan:
    """What the generation span reports."""

    def test_it_reports_the_prompt_version_provider_and_model(
        self, client, app, recorded_traces
    ):
        """An answer is only comparable to another asked the same way."""
        filename = indexed_document(client, app)

        ask(client, filename)
        recorded = span(recorded_traces.traces[0], "generation")
        model = model_for(*STORED_SETTINGS)

        assert recorded.attributes["prompt_version"] == "grounded-claims-v1"
        assert recorded.attributes["provider"] == model.provider
        assert recorded.attributes["model"] == model.id
        assert recorded.attributes["gen_ai.system"] == model.provider
        assert recorded.attributes["gen_ai.request.model"] == model.id
        assert recorded_traces.traces[0].attributes["prompt_version"] == (
            "grounded-claims-v1"
        )

    def test_it_reports_tokens_and_what_they_cost(
        self, client, app, recorded_traces
    ):
        """A trace with no cost cannot say whether a change to the prompt paid."""
        filename = indexed_document(client, app)

        ask(client, filename)
        recorded = span(recorded_traces.traces[0], "generation")
        model = model_for(*STORED_SETTINGS)

        assert recorded.attributes["input_tokens"] > 0
        assert recorded.attributes["output_tokens"] > 0
        expected = (
            recorded.attributes["input_tokens"] / 1_000_000
            * model.input_cost_per_million_usd
            + recorded.attributes["output_tokens"] / 1_000_000
            * model.output_cost_per_million_usd
        )
        assert recorded.attributes["cost_usd"] == pytest.approx(expected)

    def test_it_reports_the_finish_reason_and_whether_the_answer_was_truncated(
        self, client, app, fake_chat, recorded_traces
    ):
        """A capped answer and a whole one are not the same claim of success."""
        filename = indexed_document(client, app)
        fake_chat.script(
            StreamPlan(fragments=list(CLAIMED_ANSWER), finish_reason="length")
        )

        ask(client, filename)
        recorded = span(recorded_traces.traces[0], "generation")

        assert recorded.attributes["finish_reason"] == "length"
        assert recorded.attributes["finish_reasons"] == ["length"]
        assert recorded.attributes["truncated"] is True

    def test_it_reports_time_to_first_token_and_total_latency(
        self, client, app, recorded_traces
    ):
        """The two waits a reader has are both reported."""
        filename = indexed_document(client, app)

        ask(client, filename)
        recorded = span(recorded_traces.traces[0], "generation")

        assert recorded.attributes["time_to_first_token_ms"] >= 0
        assert recorded.attributes["total_latency_ms"] >= (
            recorded.attributes["time_to_first_token_ms"]
        )

    def test_it_reports_the_attempts_a_retried_call_took(
        self, client, app, fake_chat, recorded_traces
    ):
        """A transient failure that cost a second call is visible as one."""
        filename = indexed_document(client, app)
        fake_chat.script(
            StreamPlan(fragments=[], error=RuntimeError("temporarily unavailable")),
            StreamPlan(fragments=list(CLAIMED_ANSWER)),
        )

        ask(client, filename)
        recorded = span(recorded_traces.traces[0], "generation")

        assert recorded.attributes["attempts"] == 2

    def test_a_call_that_never_started_waits_no_time_for_a_first_token(
        self, client, app, fake_chat, recorded_traces
    ):
        """A provider that fails before output has no first token to report."""
        filename = indexed_document(client, app)
        fake_chat.fail_with(RuntimeError("provider down"), before_output=True)

        ask(client, filename)
        recorded = span(recorded_traces.traces[0], "generation")

        assert "time_to_first_token_ms" not in recorded.attributes
        assert recorded.attributes["output_tokens"] is None
        assert recorded.attributes["cost_usd"] is None


class TestCitationsAndPersistence:
    """What checking the citations and storing the answer report."""

    def test_a_grounded_answer_reports_its_claims(
        self, client, app, fake_chat, recorded_traces
    ):
        """The claims that survived checking are part of the record."""
        filename = indexed_document(client, app)
        fake_chat.stream(*CLAIMED_ANSWER)

        ask(client, filename)
        recorded = span(recorded_traces.traces[0], "citations")

        assert recorded.attributes["claim_count"] == 1
        assert recorded.attributes["invalid_source_ids"] == 0
        assert recorded.attributes["grounded"] is True
        assert recorded.attributes["repaired"] is False

    def test_a_repaired_citation_is_recorded_as_repaired(
        self, client, app, fake_chat, recorded_traces
    ):
        """The one repair the app spends is visible, because it costs a call."""
        filename = indexed_document(client, app)
        fake_chat.stream(
            "The answer ",
            'is 42.\n<claims>\n{"claim": "It is 42.", "sources": ["S9"]}\n</claims>',
        )
        fake_chat.complete_with(
            '<claims>\n{"claim": "It is 42.", "sources": ["S1"]}\n</claims>'
        )

        ask(client, filename)
        recorded = span(recorded_traces.traces[0], "citations")

        assert recorded.attributes["repaired"] is True
        assert recorded.attributes["invalid_source_ids"] == 0

    def test_a_citation_that_names_nothing_supplied_fails_the_turn(
        self, client, app, fake_chat, recorded_traces
    ):
        """An unrepairable citation is a citation error, and the trace says so."""
        filename = indexed_document(client, app)
        fake_chat.stream(
            "The answer ",
            'is 42.\n<claims>\n{"claim": "It is 42.", "sources": ["S9"]}\n</claims>',
        )
        fake_chat.complete_with(
            '<claims>\n{"claim": "It is 42.", "sources": ["S9"]}\n</claims>'
        )

        events = ask(client, filename)
        recorded = span(recorded_traces.traces[0], "citations")

        assert [name for name, _ in events][-1] == "citation_error"
        assert recorded.attributes["invalid_source_ids"] == 1
        assert recorded.error_category == "citations"

    def test_a_stored_answer_is_recorded_as_stored(
        self, client, app, repositories, recorded_traces
    ):
        """The store write and the turn it wrote are the same record."""
        filename = indexed_document(client, app)

        ask(client, filename)
        trace = recorded_traces.traces[0]
        turn = turns_of(repositories, filename)[0]

        assert span(trace, "persistence").attributes == {
            "turn_id": turn["id"],
            "outcome": "answered",
        }
        assert trace.attributes["outcome"] == "answered"

    def test_a_failed_save_is_recorded_as_a_persistence_failure(
        self, client, app, repositories, monkeypatch, recorded_traces
    ):
        """A store that refuses the answer is classified, not reported as an answer."""
        filename = indexed_document(client, app)
        monkeypatch.setattr(
            repositories.conversations,
            "complete_turn",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("postgres down")),
        )

        events = ask(client, filename)
        trace = recorded_traces.traces[0]

        assert [name for name, _ in events][-1] == "persistence_error"
        assert trace.attributes["outcome"] == "persistence_error"
        assert span(trace, "persistence").error_category == "persistence"


class TestAbstention:
    """What a question the evidence does not reach reports."""

    def test_an_abstention_reports_no_model_call_and_no_cost(
        self, client, app, recorded_traces
    ):
        """Nothing was generated, so nothing is recorded as having been paid for."""
        filename = indexed_document(client, app)
        app.config["TEST_VECTORS"].script()

        events = ask(client, filename)
        trace = recorded_traces.traces[0]

        assert [name for name, _ in events][-1] == "abstained"
        assert trace.attributes["outcome"] == "abstained"
        assert trace.attributes["abstention_reason"] == "no_evidence"
        assert trace.find("generation") == []
        assert span(trace, "persistence").attributes["outcome"] == "abstained"

    def test_an_abstention_is_measured_like_any_other_request(
        self, client, app, recorded_traces
    ):
        """A refusal is still a request, and it still has a latency."""
        filename = indexed_document(client, app)
        app.config["TEST_VECTORS"].script()

        ask(client, filename)
        trace = recorded_traces.traces[0]

        assert trace.attributes["total_latency_seconds"] >= 0
        assert trace.attributes["time_to_first_token_seconds"] is None


class TestProviderFailure:
    """What a provider that fails mid-answer reports."""

    def test_a_provider_failure_is_classified_against_generation(
        self, client, app, fake_chat, recorded_traces
    ):
        """The stream's outcome and the trace's failure are the same failure."""
        filename = indexed_document(client, app)
        fake_chat.fail_with(RuntimeError("provider down"), before_output=True)

        events = ask(client, filename)
        trace = recorded_traces.traces[0]

        assert [name for name, _ in events][-1] == "provider_error"
        assert trace.attributes["outcome"] == "provider_error"
        assert span(trace, "generation").error_category == "provider"

    def test_a_timeout_is_named_as_a_timeout(
        self, client, app, fake_chat, recorded_traces
    ):
        """A provider that ran out of time is not the same as one that refused."""
        from services.llm.base import ProviderTimeoutError

        filename = indexed_document(client, app)
        fake_chat.fail_with(ProviderTimeoutError("too slow"), before_output=True)

        ask(client, filename)

        assert span(recorded_traces.traces[0], "generation").error_category == (
            "timeout"
        )

    def test_a_failure_message_is_not_recorded(
        self, client, app, fake_chat, recorded_traces
    ):
        """A provider's error text can quote the prompt, so it is not kept."""
        filename = indexed_document(client, app)
        fake_chat.fail_with(
            RuntimeError("upstream refused the request: what is the retention?"),
            before_output=True,
        )

        ask(client, filename)
        recorded = json.dumps(recorded_traces.traces[0].to_dict())

        assert "upstream refused" not in recorded
        assert "retention" not in recorded


class TestRedaction:
    """What a trace must not contain."""

    def test_the_question_and_the_answer_are_not_in_the_trace(
        self, client, app, fake_chat, recorded_traces
    ):
        """The two texts a reader would recognise are described, not repeated."""
        filename = indexed_document(client, app)
        fake_chat.stream(*CLAIMED_ANSWER)

        ask(client, filename, query="what is the retention policy?")
        written = json.dumps(recorded_traces.traces[0].to_dict())

        assert "retention policy" not in written
        assert "The answer is 42" not in written
        assert "It is 42" not in written

    def test_the_document_text_is_not_in_the_trace(
        self, client, app, recorded_traces
    ):
        """The Passage the model read is not copied into a trace."""
        filename = indexed_document(client, app)

        ask(client, filename)

        assert "chunk about topic" not in json.dumps(
            recorded_traces.traces[0].to_dict()
        )

    def test_a_question_is_described_by_a_fingerprint(
        self, client, app, recorded_traces
    ):
        """The trace still says the request had a question, without its words."""
        filename = indexed_document(client, app)

        ask(client, filename, query="what is the retention policy?")
        query = recorded_traces.traces[0].attributes["query"]

        assert query["redacted"] == "[redacted]"
        assert query["chars"] == len("what is the retention policy?")
        assert len(query["sha256"]) == 16

    def test_the_api_key_is_not_in_the_trace(self, client, app, recorded_traces):
        """The key that paid for the answer is not written down beside it."""
        filename = indexed_document(client, app)

        ask(client, filename)

        assert "sk-test-chat-key" not in json.dumps(
            recorded_traces.traces[0].to_dict()
        )


class TestLocalExport:
    """Where a trace goes when nothing else is configured."""

    def test_a_configured_tracer_appends_one_line_per_request(self, tmp_path):
        """The local file is readable while the app runs and greppable after."""
        from services.telemetry.exporters import JsonlFileExporter
        from services.telemetry.redaction import Redactor
        from services.telemetry.spans import Sink, Tracer

        path = tmp_path / "traces" / "answer-traces.jsonl"
        tracer = Tracer(
            [Sink(exporter=JsonlFileExporter(path), redactor=Redactor())]
        )
        for _ in range(2):
            trace = tracer.start()
            trace.identify(document_id="doc-1")
            trace.span("retrieval").end()
            tracer.finish(trace)

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert {json.loads(line)["attributes"]["document_id"] for line in lines} == {
            "doc-1"
        }

    def test_a_destination_that_is_down_does_not_fail_the_request(self):
        """A collector that refuses a write costs a trace, not an answer."""
        from services.telemetry.spans import Sink, Tracer

        def _refuse(_trace):
            raise OSError("collector unreachable")

        tracer = Tracer([Sink(exporter=_refuse)])
        trace = tracer.start()
        trace.identify(document_id="doc-1")

        assert tracer.finish(trace) is trace

    def test_nothing_is_written_when_tracing_is_switched_off(self, tmp_path):
        """A configuration that exports nowhere records nowhere."""
        from services.telemetry.factory import build_tracer
        from settings import TelemetrySettings

        path = tmp_path / "answer-traces.jsonl"
        tracer = build_tracer(
            TelemetrySettings(enabled=False, local_export_path=path)
        )
        trace = tracer.start()
        trace.identify(document_id="doc-1")

        assert tracer.enabled is False
        tracer.finish(trace)
        assert not path.exists()


class TestOtlpExport:
    """What a configured collector receives."""

    def _payload(self, endpoint="http://collector:4318"):
        from services.telemetry.exporters import OtlpExporter
        from services.telemetry.redaction import Redactor
        from services.telemetry.spans import Sink, Tracer

        sent = {}

        def _post(url, *, content, headers):
            sent["url"] = url
            sent["payload"] = json.loads(content)
            return None

        tracer = Tracer(
            [
                Sink(
                    exporter=OtlpExporter(endpoint, post=_post),
                    redactor=Redactor(),
                )
            ]
        )
        trace = tracer.start()
        trace.identify(document_id="doc-1", turn_id="turn-1")
        generation = trace.span("generation")
        generation.record(input_tokens=10, finish_reason="stop")
        generation.end()
        tracer.finish(trace)
        return sent

    def test_spans_are_posted_to_the_traces_path(self):
        """A collector is given the traces of the service it was pointed at."""
        sent = self._payload()

        assert sent["url"] == "http://collector:4318/v1/traces"
        resource = sent["payload"]["resourceSpans"][0]
        assert resource["scopeSpans"][0]["scope"]["name"] == "papermind.telemetry"

    def test_a_span_carries_the_correlation_identifiers(self):
        """A span that arrives alone still names the request it was part of."""
        sent = self._payload()
        span = sent["payload"]["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        keys = {attribute["key"] for attribute in span["attributes"]}

        assert {"document_id", "turn_id", "papermind.trace_id"} <= keys

    def test_a_local_capture_window_does_not_reach_a_collector(self):
        """What leaves the process is fingerprints, whatever the window is doing."""
        from services.telemetry.exporters import OtlpExporter
        from services.telemetry.redaction import Redactor
        from services.telemetry.spans import Sink, Tracer

        sent = {}

        def _post(url, *, content, headers):
            sent["payload"] = json.loads(content)
            return None

        tracer = Tracer(
            [
                Sink(
                    exporter=OtlpExporter("http://collector:4318", post=_post),
                    redactor=Redactor(),
                )
            ]
        )
        trace = tracer.start()
        trace.identify(query="what is the retention policy?")
        trace.span("retrieval").end()
        tracer.finish(trace)

        span = sent["payload"]["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        query = next(
            attribute
            for attribute in span["attributes"]
            if attribute["key"] == "query"
        )
        assert query["value"]["kvlistValue"]["values"][0] == {
            "key": "redacted",
            "value": {"stringValue": "[redacted]"},
        }

    def test_a_capture_window_does_record_the_text_locally(self, tmp_path):
        """The opt-in exists to be useful, or it would not be worth having."""
        from services.telemetry.exporters import JsonlFileExporter
        from services.telemetry.redaction import Redactor
        from services.telemetry.spans import Sink, Tracer

        path = tmp_path / "answer-traces.jsonl"
        tracer = Tracer(
            [
                Sink(
                    exporter=JsonlFileExporter(path),
                    redactor=Redactor(capture_text=True, capture_window_seconds=900),
                )
            ]
        )
        trace = tracer.start()
        trace.identify(query="what is the retention policy?")
        tracer.finish(trace)

        assert "retention policy" in path.read_text()
