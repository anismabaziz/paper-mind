"""
One number a reviewer can act on: what it cost in quality, seconds, and dollars.

Three questions are usually answered by three documents, and each of them can be
flattered by the other two being absent. A quality report drops the cases that
failed, a latency report drops the cases that were hard, and a cost report drops
everything the provider was not billed for. This module is the one place they are
answered together, from the same cases, so that a configuration cannot look cheap
by being asked fewer questions or look fast by failing the slow ones.

Two sources feed it and neither is optional to the other. The evaluation run
says what the agreed case set produced; the recorded traces say what readers
actually waited for. Where a run had no trace to read, the report says so rather
than leaving the cell empty, because an empty cell reads as a measurement that
found nothing.
"""

import json

import pytest

from evaluation import digest, report
from evaluation.dataset import REPORTED
from tests.test_evaluation_report import MODEL, a_manifest, a_result, dataset

# The committed case set, defined once in the report's own tests.
__all__ = ["dataset"]

ANSWER_METRICS = (
    "correctness",
    "faithfulness",
    "citation_precision",
    "citation_recall",
)


def a_case(case_id, outcome="answered", failed=(), category="exact_lookup"):
    """Return one measured case, as a result file stores it."""
    return {
        "id": case_id,
        "category": category,
        "outcome": outcome,
        "grades": {
            name: {"outcome": "failed", "detail": "no exact evidence"}
            for name in failed
        },
        "hit_at_k": outcome == "answered",
        "ndcg_at_k": 1.0 if outcome == "answered" else 0.0,
    }


def an_answer_result(**changes):
    """Return the result of an answer-path run, with one number per cell."""
    answers = {
        "generated": 3,
        "truncated": 0,
        "finish_reasons": {"stop": 3},
        **{
            name: {
                "graded": 3,
                "scored": 3,
                "mean": 0.8,
                "passed": 2,
                "failed": 1,
                "unknown": 0,
            }
            for name in (*ANSWER_METRICS, "abstention")
        },
    }
    result = {
        "experiment": "baseline",
        "family": "shipped",
        "dataset": "2026-09-labeled-cases-v1",
        "split": REPORTED,
        "held_back": [],
        "index": {},
        "retrieval": {
            "questions": 3,
            "k": 5,
            "hit_rate": 0.9,
            "recall": 0.7,
            "mrr": 0.6,
            "ndcg": 0.75,
        },
        "answers": answers,
        "outcomes": {
            "answered": 2,
            "abstained": 1,
            "provider_error": 0,
            "citation_error": 0,
            "persistence_error": 0,
            "refused": 0,
            "cancelled": 0,
        },
        "latency": {
            "cold_start": {
                name: {"samples": 1, "p50": 4.0, "p95": 4.0}
                for name in (
                    "retrieval_seconds",
                    "first_token_seconds",
                    "total_seconds",
                )
            },
            "steady_state": {
                name: {"samples": 2, "p50": 0.5, "p95": 0.9}
                for name in (
                    "retrieval_seconds",
                    "first_token_seconds",
                    "total_seconds",
                )
            },
            "retrieval_seconds": {"samples": 2, "p50": 0.5, "p95": 0.9},
        },
        "cost": {
            "input_tokens": 12000,
            "output_tokens": 900,
            "usd": 0.05,
            "priced_cases": 3,
            "model": "qwen/qwen3.8-27b",
            "input_cost_per_million_usd": 0.8,
            "output_cost_per_million_usd": 4.0,
        },
        "calibration": None,
        "judge": None,
        "cases": [
            a_case("notes-stages"),
            a_case("notes-hit-rate", failed=["exact_evidence"]),
            a_case("notes-abstain", outcome="abstained", category="unanswerable"),
        ],
        "evidence": {
            kind: {"measured": True, "reason": "measured by this run"}
            for kind in report.EVIDENCE
        },
        **changes,
    }
    return result


def a_digest(dataset, settings_obj, results=None, traces=None):
    """Return the digest of an answer-path run, built the way the CLI builds it."""
    results = [an_answer_result()] if results is None else results
    return digest.digest(
        a_manifest(dataset, settings_obj, results, run_records=[]),
        results,
        traces=traces,
    )


@pytest.fixture
def trace_files(tmp_path):
    """Return two export files holding five recorded requests between them."""
    from tests.test_evaluation_traces import a_trace, write

    return [
        write(
            tmp_path / "traces-1.jsonl",
            a_trace(total=1.0),
            a_trace(total=1.0),
            a_trace(total=9.0),
        ),
        write(tmp_path / "traces-2.jsonl", a_trace(total=1.0), a_trace(total=1.0)),
    ]


@pytest.fixture
def measured(dataset, settings_obj):
    """Return the digest of one answer-path run."""
    return a_digest(dataset, settings_obj)


class TestQuality:
    """What the run was worth, broken down by the thing each grade decided."""

    def test_it_separates_retrieval_from_answers_citations_and_abstention(
        self, measured
    ):
        """A retrieval mean and an answer mean are not comparable and not merged."""
        quality = measured["quality"]

        assert quality["retrieval"] == {
            "questions": 3,
            "hit_rate": 0.9,
            "recall": 0.7,
            "mrr": 0.6,
            "ndcg": 0.75,
        }
        assert quality["answers"]["correctness"]["mean"] == 0.8
        assert quality["citations"]["citation_precision"]["mean"] == 0.8
        assert quality["abstention"]["mean"] == 0.8

    def test_it_reports_the_unknown_cases_beside_every_mean(self, measured):
        """A mean over the cases that survived is better than it looks."""
        quality = measured["quality"]

        assert quality["answers"]["correctness"]["scored"] == 3
        assert quality["answers"]["correctness"]["graded"] == 3
        assert quality["answers"]["correctness"]["unknown"] == 0

    def test_it_breaks_failures_down_by_the_outcome_and_the_grader_that_failed(
        self, measured
    ):
        """A cheap configuration must not look cheap by dropping its failures."""
        quality = measured["quality"]

        assert quality["failures"]["outcomes"] == {"answered": 2, "abstained": 1}
        assert quality["failures"]["graders"] == {"exact_evidence": 1}
        assert quality["failures"]["cases"] == ["notes-hit-rate", "notes-abstain"]

    def test_an_abstention_is_counted_as_a_result_not_as_a_missing_answer(
        self, measured
    ):
        """Refusing to answer is a decision the run made, and it has a count."""
        assert measured["quality"]["failures"]["outcomes"]["abstained"] == 1
        assert measured["quality"]["abstention"]["graded"] == 3


class TestAConfigurationTableOverRetrievalOnlyResults:
    """A retrieval report's columns are retrieval numbers and retrieval seconds."""

    def test_it_compares_the_retrieval_latency_a_retrieval_run_measured(
        self, dataset, settings_obj
    ):
        """A variant that answers the same question faster is the whole point."""
        cheap = a_result()
        slow = a_result(
            "dense-only",
            latency={"retrieval_seconds": {"samples": 2, "p50": 0.02, "p95": 0.03}},
        )

        rows = digest.compare([cheap, cheap], [cheap, slow])

        assert rows[1]["experiment"] == "dense-only"
        assert rows[1]["latency"]["retrieval_seconds_p50"] == pytest.approx(0.01)
        assert rows[1]["latency"]["retrieval_seconds_p95"] == pytest.approx(0.01)
        assert rows[1]["latency"]["total_seconds_p50"] is None


class TestAReportThatMeasuredOnlyRetrieval:
    """A retrieval-only run has half a report, and has to say which half."""

    def test_it_names_the_kinds_it_never_measured_and_why(self, dataset, settings_obj):
        """A row of dashes reads as a measurement that found nothing."""
        measured = a_digest(dataset, settings_obj, [a_result()])

        assert measured["quality"]["unmeasured"]["answers"] == (
            "retrieval-only measurement, nothing asked a model"
        )
        assert measured["quality"]["answers"] == {}
        assert measured["quality"]["abstention"] is None

    def test_its_printed_quality_says_what_it_never_measured(
        self, dataset, settings_obj
    ):
        """The prose has to carry the reason, not just the numbers below it."""
        written = digest.render(a_digest(dataset, settings_obj, [a_result()]))

        assert "not measured" in written
        assert "nothing asked a model" in written

    def test_it_reports_the_retrieval_latency_a_retrieval_run_did_measure(
        self, dataset, settings_obj
    ):
        """A retrieval-only report still knows how long retrieval took."""
        result = a_result()
        result["latency"] = {
            "retrieval_seconds": {"samples": 2, "p50": 0.01, "p95": 0.02}
        }

        measured = a_digest(dataset, settings_obj, [result])

        steady = measured["performance"]["evaluation"]["steady_state"]
        assert steady["retrieval_seconds"] == {"samples": 2, "p50": 0.01, "p95": 0.02}
        assert steady["total_seconds"] == {}


class TestPerformance:
    """What a reader waited, and what a run of them could carry."""

    def test_it_reports_p50_and_p95_for_every_wait_a_reader_has(
        self, dataset, settings_obj
    ):
        """Retrieval, first word, and the whole request are three numbers."""
        result = an_answer_result()
        result["latency"]["steady_state"]["first_token_seconds"] = {
            "samples": 2,
            "p50": 1.5,
            "p95": 2.5,
        }
        result["latency"]["steady_state"]["total_seconds"] = {
            "samples": 2,
            "p50": 3.0,
            "p95": 6.0,
        }

        measured = a_digest(dataset, settings_obj, [result])

        assert measured["performance"]["evaluation"]["steady_state"] == {
            "retrieval_seconds": {"samples": 2, "p50": 0.5, "p95": 0.9},
            "first_token_seconds": {"samples": 2, "p50": 1.5, "p95": 2.5},
            "total_seconds": {"samples": 2, "p50": 3.0, "p95": 6.0},
        }

    def test_it_holds_the_cold_start_apart_from_the_steady_state(self, measured):
        """A reader's first question pays for loading, and nobody pays it twice."""
        performance = measured["performance"]["evaluation"]

        assert performance["cold_start"]["total_seconds"]["p50"] == 4.0
        assert performance["steady_state"]["total_seconds"]["p50"] == 0.5

    def test_it_reports_throughput_from_the_recorded_traces(
        self, dataset, settings_obj, trace_files
    ):
        """A rate of arrival is not a rate of seconds per call."""
        measured = a_digest(
            dataset, settings_obj, traces=[trace_files[0], trace_files[1]]
        )

        assert measured["performance"]["traces"]["traces"] == 5
        assert (
            measured["performance"]["traces"]["throughput"]["traces_per_minute"]
            == 33.333
        )
        assert (
            measured["performance"]["traces"]["latency"]["total_seconds"]["p95"] == 7.4
        )

    def test_it_says_it_had_no_traces_rather_than_showing_an_empty_cell(self, measured):
        """An empty cell reads as a measurement that found nothing."""
        assert measured["performance"]["traces"] == {
            "measured": False,
            "reason": "no recorded traces were read for this report",
        }


class TestCost:
    """What a run cost, in tokens, in dollars, and in the reader's own seconds."""

    def test_it_reports_the_provider_tokens_and_price_the_run_paid(self, measured):
        """Cost is tokens at a named price, not a total somebody rounded."""
        cost = measured["cost"]["provider"]

        assert cost["input_tokens"] == 12000
        assert cost["output_tokens"] == 900
        assert cost["usd"] == 0.05
        assert cost["per_case_usd"] == 0.016667
        assert cost["priced_cases"] == 3
        assert cost["model"] == "qwen/qwen3.8-27b"

    def test_it_states_the_local_compute_the_embedding_and_rerank_ran_on(
        self, dataset, settings_obj
    ):
        """The local half of the cost is somebody's CPU, and says which."""
        result = an_answer_result(
            index={"notes.pdf": {"index_generation": 2, "indexed_seconds": 12.0}}
        )
        manifest = a_manifest(dataset, settings_obj, [result], run_records=[])

        local = digest.digest(manifest, [result])["cost"]["local"]

        assert local["device"] == "cpu"
        assert local["billed_usd"] == 0.0
        assert local["embedding_model"] == manifest["models"]["embedding"]["id"]
        assert local["rerank_model"] == manifest["models"]["reranker"]["id"]
        assert local["rerank_applied"] is False
        assert local["documents"] == 1
        assert local["indexed_seconds"] == 12.0
        assert local["seconds_per_document"] == 12.0

    def test_it_reports_the_traces_own_cost_and_tokens_next_to_the_runs(
        self, dataset, settings_obj, trace_files
    ):
        """Readers outside the case set spent money too, and the tokens behind it."""
        measured = a_digest(dataset, settings_obj, traces=[trace_files[0]])

        assert measured["cost"]["traces"]["usd"] == 0.0039
        assert measured["cost"]["traces"]["input_tokens"] == 2700
        assert measured["cost"]["traces"]["output_tokens"] == 450
        assert measured["cost"]["traces"]["traces"] == 3
        assert measured["cost"]["traces"]["model"] == "qwen/qwen3.8-27b"


class TestWhatTheDigestWasMeasuredOn:
    """The versions a number is only comparable to another number under."""

    def test_it_names_the_case_set_the_revision_and_the_prompt(
        self, dataset, settings_obj, measured
    ):
        """A number without these cannot be reproduced or compared."""
        provenance = measured["provenance"]

        assert provenance["dataset"] == dataset.version
        assert provenance["split"] == REPORTED
        assert provenance["revision"] == "0" * 40
        assert provenance["citation_prompt"]
        assert provenance["judge_rubric"] is None

    def test_it_names_the_generator_and_judge_and_their_revisions(
        self, dataset, settings_obj
    ):
        """A model revision is the one thing a checkout cannot pin."""
        result = an_answer_result(
            judge={"provider": "groq", "model": "openai/gpt-oss-20b", "rubric": "v2"}
        )
        manifest = a_manifest(
            dataset,
            settings_obj,
            [result],
            run_records=[
                {
                    "run": {
                        "provider": "groq",
                        "model": "qwen/qwen3.8-27b",
                        "judge": {
                            "provider": "groq",
                            "model": "openai/gpt-oss-20b",
                            "rubric_version": "v2",
                        },
                    }
                }
            ],
        )

        provenance = digest.digest(manifest, [result])["provenance"]

        assert provenance["generator"] == {
            "provider": "groq",
            "id": "qwen/qwen3.8-27b",
        }
        assert provenance["judge"]["id"] == "openai/gpt-oss-20b"
        assert provenance["judge_rubric"] == "v2"

    def test_it_records_the_index_generation_every_document_was_answered_from(
        self, dataset, settings_obj
    ):
        """An answer read from a different generation is a different measurement."""
        result = an_answer_result(
            index={
                "notes.pdf": {"index_generation": 3, "indexed_seconds": 1.0},
                "primer.pdf": {"index_generation": 1, "indexed_seconds": 2.0},
            }
        )
        manifest = a_manifest(dataset, settings_obj, [result], run_records=[])

        index = digest.digest(manifest, [result])["provenance"]["index"]

        assert index == {
            "notes.pdf": 3,
            "primer.pdf": 1,
        }

    def test_it_carries_nothing_a_reader_typed_or_a_key(self, measured):
        """A committed digest must be safe to publish."""
        written = json.dumps(measured)

        assert "api_key" not in written
        assert "PAPERMIND_EVAL" not in written


class TestTwoReportsSideBySide:
    """Every configuration with what it gained and what it cost."""

    def test_it_reports_the_quality_change_and_the_runtime_cost_together(
        self, dataset, settings_obj
    ):
        """A variant that scores better and costs more is a decision, not a result."""
        cheap = an_answer_result()
        rich = an_answer_result(
            experiment="dense-only",
            answers={
                **cheap["answers"],
                "correctness": {**cheap["answers"]["correctness"], "mean": 0.9},
            },
            latency={
                **cheap["latency"],
                "steady_state": {
                    name: {"samples": 2, "p50": 0.9, "p95": 1.4}
                    for name in (
                        "retrieval_seconds",
                        "first_token_seconds",
                        "total_seconds",
                    )
                },
            },
            cost={**cheap["cost"], "usd": 0.09},
        )

        rows = digest.compare([cheap], [rich])

        assert len(rows) == 1
        assert rows[0]["experiment"] == "dense-only"
        assert rows[0]["quality"]["ndcg"] == 0.0
        assert rows[0]["quality"]["correctness"] == pytest.approx(0.1)
        assert rows[0]["latency"]["total_seconds_p50"] == 0.4
        assert rows[0]["cost"]["usd"] == pytest.approx(0.04)
        assert rows[0]["cost"]["per_case_usd"] == pytest.approx(0.013333)

    def test_it_keeps_the_failures_of_both_sides_in_the_row(
        self, dataset, settings_obj
    ):
        """A variant that answered fewer questions must not read as a better one."""
        cheap = an_answer_result()
        fewer = an_answer_result(
            experiment="dense-only",
            cases=[
                a_case("notes-stages"),
                a_case("notes-hit-rate", failed=["exact_evidence"]),
                a_case("notes-abstain", outcome="provider_error"),
            ],
            outcomes={**cheap["outcomes"], "abstained": 0, "provider_error": 1},
        )

        row = digest.compare([cheap], [fewer])[0]

        assert row["failures"]["before"] == {"answered": 2, "abstained": 1}
        assert row["failures"]["after"] == {"answered": 2, "provider_error": 1}
        assert row["cases"]["before"] == 3
        assert row["cases"]["after"] == 3

    def test_it_compares_a_configuration_against_the_shipped_one(
        self, dataset, settings_obj
    ):
        """A variant has nothing to be better than until it has a baseline."""
        cheap = an_answer_result()
        variant = an_answer_result(experiment="dense-only")

        rows = digest.compare([cheap, variant], [cheap, variant])

        # The shipped configuration is compared with itself, so every other
        # column's differences read as differences rather than as absolutes.
        assert [row["experiment"] for row in rows] == ["baseline", "dense-only"]
        assert rows[0]["quality"]["ndcg"] == 0.0
        assert rows[1]["quality"]["ndcg"] == 0.0


class TestTheConfigurationTable:
    """Every configuration that was measured, with what it gained and cost."""

    def test_a_report_of_several_configurations_compares_them_all(
        self, dataset, settings_obj
    ):
        """A report with eleven experiments that shows one of them is not a report."""
        cheap = an_answer_result()
        rich = an_answer_result(
            experiment="dense-only",
            answers={
                **cheap["answers"],
                "correctness": {**cheap["answers"]["correctness"], "mean": 0.9},
            },
            cost={**cheap["cost"], "usd": 0.09},
        )

        measured = a_digest(dataset, settings_obj, [cheap, rich])

        assert [row["experiment"] for row in measured["configurations"]] == [
            "baseline",
            "dense-only",
        ]
        assert measured["configurations"][1]["quality"]["correctness"] == pytest.approx(
            0.1
        )
        assert measured["configurations"][1]["cost"]["usd"] == pytest.approx(0.04)

    def test_one_configuration_has_nothing_to_be_compared_with(
        self, dataset, settings_obj
    ):
        """A single-column comparison is a row of zeroes, and says nothing."""
        measured = a_digest(dataset, settings_obj)

        assert measured["configurations"] == []

    def test_the_table_shows_the_retrieval_seconds_a_variant_changed(
        self, dataset, settings_obj
    ):
        """A variant that ranks better and retrieves slower is a trade, not a win."""
        cheap = a_result()
        slow = a_result(
            "dense-only",
            latency={"retrieval_seconds": {"samples": 2, "p50": 0.02, "p95": 0.03}},
        )

        written = digest.render(a_digest(dataset, settings_obj, [cheap, slow]))

        assert "Retrieval p50 (s)" in written
        row = next(
            line for line in written.splitlines() if line.startswith("| `dense-only`")
        )
        assert row.split(" | ")[3] == "+0.010"

    def test_a_variant_that_reindexes_shows_the_seconds_it_cost(
        self, dataset, settings_obj
    ):
        """A chunking variant re-embeds every document, and that is its price."""
        cheap = a_result(
            index={"a.pdf": {"index_generation": 1, "indexed_seconds": 4.0}}
        )
        rechunked = a_result(
            "chunk-256-25",
            index={"a.pdf": {"index_generation": 2, "indexed_seconds": 12.0}},
        )

        rows = digest.compare([cheap, cheap], [cheap, rechunked])

        assert rows[1]["cost"]["indexed_seconds"] == pytest.approx(8.0)

    def test_a_run_that_billed_nothing_says_so_rather_than_reporting_zero(
        self, dataset, settings_obj
    ):
        """Zero tokens and a model called zero times are not the same claim."""
        written = digest.render(a_digest(dataset, settings_obj, [a_result()]))

        assert "billed nothing" in written

    def test_the_report_prints_the_comparison_with_both_halves(
        self, dataset, settings_obj
    ):
        """A reader deciding on a variant needs the gain and the price in one table."""
        cheap = an_answer_result()
        rich = an_answer_result(
            experiment="dense-only", cost={**cheap["cost"], "usd": 0.09}
        )
        measured = a_digest(dataset, settings_obj, [cheap, rich])

        written = digest.render(measured)

        assert "## Configurations" in written
        assert "dense-only" in written
        assert "nDCG" in written or "ndcg" in written


class TestThresholds:
    """A CI gate that only fires on a regression somebody declared."""

    def test_it_reports_the_thresholds_a_run_crossed(self, measured):
        """Failing on everything is the same as failing on nothing."""
        crossed = digest.regressions(
            measured,
            {
                "quality.retrieval.hit_rate": {"min": 0.95},
                "quality.answers.correctness.mean": {"min": 0.75},
                "cost.provider.usd": {"max": 0.10},
            },
        )

        assert [row["threshold"] for row in crossed] == ["quality.retrieval.hit_rate"]
        assert crossed[0]["bound"] == 0.95
        assert crossed[0]["direction"] == "min"

    def test_a_threshold_it_cannot_measure_is_a_failure_to_measure_not_a_pass(
        self, measured
    ):
        """A gate that skips what it cannot read is a gate nobody is watching."""
        crossed = digest.regressions(measured, {"performance.traces.p95": {"max": 1}})

        assert [row["threshold"] for row in crossed] == ["performance.traces.p95"]
        assert crossed[0]["reason"] == "not measured by this run"

    def test_a_run_inside_every_threshold_crosses_nothing(self, measured):
        """Nothing to say is the good outcome, and has to be possible."""
        assert (
            digest.regressions(
                measured,
                {
                    "quality.retrieval.hit_rate": {"min": 0.5},
                    "cost.provider.usd": {"max": 1.0},
                },
            )
            == []
        )

    def test_a_latency_threshold_names_the_population_it_bounds(self, measured):
        """A bound with no population in it is a bound on whichever one is read first."""
        crossed = digest.regressions(
            measured,
            {"performance.evaluation.steady_state.total_seconds.p95": {"max": 0.8}},
        )

        assert [row["threshold"] for row in crossed] == [
            "performance.evaluation.steady_state.total_seconds.p95"
        ]
        # The steady state's 0.9, not the cold start's 4.0: the cold start is one
        # case paying a lazy load, and a gate that watched it would fire whenever
        # a machine happened to be cold.
        assert crossed[0]["measured"] == 0.9
        assert crossed[0]["reason"] == "above the declared max of 0.8"

    def test_a_bound_on_the_cold_start_is_readable_as_one(self, measured):
        """Naming the cold start is possible; it is just a different question."""
        assert (
            digest.regressions(
                measured,
                {"performance.evaluation.cold_start.total_seconds.p95": {"max": 1.0}},
            )[0]["measured"]
            == 4.0
        )


class TestRenderingTheDigest:
    """The markdown a reviewer reads instead of a directory of JSON."""

    def test_it_puts_quality_latency_and_cost_in_one_table(self, measured):
        """Three documents can each be right and together still mislead."""
        written = digest.render(measured)

        assert "## Quality" in written
        assert "## Performance" in written
        assert "## Cost" in written
        assert "p95" in written

    def test_it_reports_what_the_recorded_requests_ended_as(
        self, dataset, settings_obj, tmp_path
    ):
        """29 recorded requests failed, and a latency table that hides that lies."""
        from tests.test_evaluation_traces import a_trace, write

        path = write(
            tmp_path / "reader-traces.jsonl",
            a_trace(),
            a_trace(),
            a_trace(),
            a_trace(outcome="provider_error"),
        )
        measured = a_digest(dataset, settings_obj, traces=[path])

        written = digest.render(measured)

        assert (
            "What those 4 requests ended as: answered 3, provider_error 1." in written
        )

    def test_it_names_what_the_run_did_not_measure(self, measured):
        """A missing trace file is a fact about the report, not a blank cell."""
        written = digest.render(measured)

        assert "no recorded traces" in written

    def test_it_prints_the_commands_that_reproduce_it(self, measured):
        """A number nobody can regenerate is an anecdote."""
        written = digest.render(measured)

        assert "python -m evaluation.cli" in written

    def test_a_retrieval_report_reproduces_itself_with_the_command_that_made_it(
        self, dataset, settings_obj
    ):
        """A command naming a model that never ran is not the command that ran."""
        written = digest.render(a_digest(dataset, settings_obj, [a_result()]))

        assert "--ablate" in written
        assert "--live \\" not in written
        assert "PAPERMIND_EVAL_GENERATOR_API_KEY" not in written

    def test_it_never_asks_for_a_trace_file_a_reader_will_not_have(
        self, dataset, settings_obj, trace_files
    ):
        """The raw export is an operator's local file; the report keeps its summary."""
        written = digest.render(
            a_digest(dataset, settings_obj, traces=[trace_files[0]])
        )

        assert "--traces" not in written
        assert "--render <directory>" in written

    def test_the_reference_row_is_labelled_as_the_reference(
        self, dataset, settings_obj
    ):
        """A row of zeroes against itself reads as a result rather than as a ruler."""
        cheap = a_result()
        variant = a_result("dense-only")

        written = digest.render(a_digest(dataset, settings_obj, [cheap, variant]))

        assert "`baseline (reference)`" in written

    def test_it_writes_the_report_next_to_the_numbers_it_describes(
        self, tmp_path, measured
    ):
        """The digest belongs in the report directory, not in someone's notes."""
        written = digest.write(tmp_path, measured)

        assert (tmp_path / "digest.md").read_text() == written
        assert json.loads((tmp_path / "digest.json").read_text()) == measured
