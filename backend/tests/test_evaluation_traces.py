"""
What the recorded traces of real requests say, summarized for a report.

An evaluation run measures the case set a reviewer agreed to ask. The traces are
what the application wrote while serving readers, and they are the half nobody
has to take on trust. A report that quotes the first without the second is
describing a script; one that quotes the second without the first cannot say
whether the run was representative. Read together they say whether the measured
run describes what a reader actually waited for.

The traces are read as they were written: one JSON object per line, the fields
the answer path records, no text. A summary that needed the question or the
answer to compute would be a summary that could not be published.
"""

import json

from services.accounts.chat_settings_service import model_for
from services.telemetry.answer import AnswerTrace
from services.telemetry.exporters import JsonlFileExporter
from services.telemetry.redaction import Redactor
from services.telemetry.spans import Sink, Tracer

from evaluation import traces

MODEL = model_for("groq", "qwen/qwen3.8-27b")


def a_trace(
    *,
    outcome="answered",
    total=1.0,
    first_token=0.4,
    retrieval=0.3,
    input_tokens=900,
    output_tokens=150,
    cost=0.0013,
    started_at=1000.0,
    model="qwen/qwen3.8-27b",
):
    """Return one recorded answer request, shaped the way the exporter writes it."""
    generation = {
        "name": "generation",
        "span_id": "aa11",
        "error_category": None,
        "started_at": started_at + retrieval,
        "ended_at": started_at + total,
        "duration_ms": (total - retrieval) * 1000,
        "attributes": {
            "provider": "groq",
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
            "finish_reason": "stop",
        },
    }
    return {
        "name": "answer.request",
        "trace_id": "5f5ec454",
        "attributes": {
            "provider": "groq",
            "model": model,
            "outcome": outcome,
            "total_latency_seconds": total,
            "time_to_first_token_seconds": first_token,
        },
        "spans": [
            {
                "name": "retrieval",
                "span_id": "bb22",
                "error_category": None,
                "started_at": started_at,
                "ended_at": started_at + retrieval,
                "duration_ms": retrieval * 1000,
                "attributes": {
                    "retrieval_method": "hybrid",
                    "retrieval_outcome": "success",
                },
            },
            generation,
        ],
    }


def write(path, *records):
    """Write traces to an export file, one JSON object per line."""
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return path


class TestWhatTracesMeasure:
    """A summary of recorded requests, in the shape a report reads."""

    def test_it_reports_how_long_answers_took_and_when_the_first_word_arrived(
        self, tmp_path
    ):
        """A reader's wait and a reader's patience are two different numbers."""
        path = write(tmp_path / "traces.jsonl", a_trace())

        summary = traces.summarize([path])

        assert summary["traces"] == 1
        assert summary["latency"]["total_seconds"] == {
            "samples": 1,
            "p50": 1.0,
            "p95": 1.0,
        }
        assert summary["latency"]["first_token_seconds"]["p50"] == 0.4
        assert summary["latency"]["retrieval_seconds"]["p50"] == 0.3

    def test_it_summarizes_the_tail_separately_from_the_median(self, tmp_path):
        """One slow request is what p95 is for, and averaging it away hides that."""
        path = write(
            tmp_path / "traces.jsonl",
            a_trace(total=1.0),
            a_trace(total=1.0),
            a_trace(total=9.0),
        )

        summary = traces.summarize([path])

        assert summary["latency"]["total_seconds"]["p50"] == 1.0
        assert summary["latency"]["total_seconds"]["p95"] == 8.2

    def test_it_prices_the_model_calls_the_traces_recorded(self, tmp_path):
        """Cost is the sum of the calls a reader made, at the prices recorded."""
        path = write(
            tmp_path / "traces.jsonl",
            a_trace(cost=0.0013, input_tokens=900, output_tokens=150),
            a_trace(cost=0.0027, input_tokens=900, output_tokens=150),
        )

        summary = traces.summarize([path])

        assert summary["cost"]["usd"] == 0.004
        assert summary["cost"]["per_trace_usd"] == 0.002
        assert summary["cost"]["priced_calls"] == 2
        assert summary["cost"]["model"] == "qwen/qwen3.8-27b"
        assert summary["tokens"] == {"input": 1800, "output": 300}

    def test_it_counts_what_each_request_ended_as(self, tmp_path):
        """A cheap configuration must not look cheap by dropping its failures."""
        path = write(
            tmp_path / "traces.jsonl",
            a_trace(),
            a_trace(),
            a_trace(),
            a_trace(outcome="provider_error"),
            a_trace(outcome="abstained"),
        )

        summary = traces.summarize([path])

        assert summary["traces"] == 5
        assert summary["outcomes"] == {
            "answered": 3,
            "provider_error": 1,
            "abstained": 1,
        }

    def test_it_reports_how_many_answers_a_reader_got_per_minute_of_wall_clock(
        self, tmp_path
    ):
        """Throughput is about the reader arriving, not about the seconds a call took."""
        path = write(
            tmp_path / "traces.jsonl",
            a_trace(total=0.3, started_at=1000.0),
            a_trace(total=0.3, started_at=1001.0),
            a_trace(total=0.3, started_at=1002.0),
            a_trace(total=0.3, started_at=1003.0),
        )

        summary = traces.summarize([path])

        assert summary["throughput"] == {
            "traces_per_minute": 72.727,
            "window_seconds": 3.3,
        }

    def test_it_reads_every_file_as_one_population(self, tmp_path):
        """Two exports from one afternoon are one set of readers, not two runs."""
        morning = write(tmp_path / "morning.jsonl", a_trace(started_at=1000.0))
        afternoon = write(tmp_path / "afternoon.jsonl", a_trace(started_at=1100.0))

        summary = traces.summarize([morning, afternoon])

        assert summary["traces"] == 2
        assert summary["outcomes"] == {"answered": 2}


class TestTracesTheApplicationWrote:
    """A summary of the file the answer path actually exports."""

    def test_it_reads_a_trace_the_answer_path_recorded(self, tmp_path):
        """The report's second half is the record the application keeps in production."""
        path = tmp_path / "answer-traces.jsonl"
        exporter = JsonlFileExporter(path)
        moments = iter([0.0, 0.5, 0.9, 0.9])
        tracer = Tracer([Sink(exporter=exporter, redactor=Redactor())])
        trace = AnswerTrace.start(tracer, model=MODEL)
        trace.identify(document_id="doc-1", turn_id="turn-1", index_generation=1)
        trace.begin(lambda: next(moments, 0.0))
        with trace.retrieval(document_id="doc-1", index_generation=1) as span:
            span.record(retrieval_method="hybrid", retrieval_outcome="success")
        with trace.generation(query="q", context="c", prior_turns="") as span:
            trace.mark_first_token()
            trace.generated(
                span,
                query="q",
                context="c",
                prior_turns="",
                generated="the retention policy is ninety days",
                finish_reason="stop",
                attempts=1,
                truncated=False,
            )
        trace.citations(claims=1, invalid_ids=0, grounded=True, repaired=False)
        trace.persistence("answered", turn_id="turn-1")
        trace.timings()
        trace.finish("answered")

        summary = traces.summarize([path])

        assert summary["traces"] == 1
        assert summary["outcomes"] == {"answered": 1}
        assert summary["latency"]["total_seconds"]["p50"] == 0.9
        assert summary["latency"]["first_token_seconds"]["p50"] == 0.5
        assert summary["cost"]["priced_calls"] == 1
        assert summary["tokens"]["input"] > 0

    def test_it_never_carries_the_question_or_the_answer_a_reader_exchanged(
        self, tmp_path
    ):
        """A committed summary must not hold what a reader typed or was told."""
        path = write(tmp_path / "traces.jsonl", a_trace())
        written = json.dumps(traces.summarize([path])).lower()

        assert "the answer" not in written
        assert "query" not in written
        assert "prompt" not in written


class TestATraceFileThatIsNotThere:
    """What a summary says when it has nothing to read."""

    def test_an_empty_file_summarizes_to_nothing_measured(self, tmp_path):
        """No samples has no p50, and a 0.0 would read as the fastest ever."""
        path = tmp_path / "traces.jsonl"
        path.write_text("", encoding="utf-8")

        summary = traces.summarize([path])

        assert summary["traces"] == 0
        assert summary["latency"]["total_seconds"]["p50"] is None
        assert summary["throughput"]["traces_per_minute"] is None
        assert summary["cost"]["usd"] == 0.0

    def test_a_torn_line_is_skipped_rather_than_lost(self, tmp_path):
        """An export file is appended to while the app runs, so the last line tears."""
        path = tmp_path / "traces.jsonl"
        path.write_text(
            json.dumps(a_trace()) + "\n" + '{"attributes": {"total_lat',
            encoding="utf-8",
        )

        summary = traces.summarize([path])

        assert summary["traces"] == 1
