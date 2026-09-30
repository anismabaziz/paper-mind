"""
Rendering the digest offline, from a report already on disk.

A live run indexes real documents and answers them with a paid provider, so it
only ever happens on purpose, and a reviewer's second look at a report should not
cost a model call to get. These tests cover the path that answers a question
about a checked-in report with no key, no provider, and no index: read the
manifest and the results, read the recorded traces beside them, and write the
one document that puts quality, latency, and cost next to each other.

The threshold gate is the other half. CI cannot afford a live run either, so it
re-renders the committed report and asks whether anything crossed a bound a
person wrote down. Anything in CI that fails on a number somebody found by eye
is a number that will be allowed to drift.
"""

import json
from pathlib import Path

import pytest

from evaluation import cli, digest, report
from tests.test_evaluation_digest import an_answer_result, a_manifest
from tests.test_evaluation_report import a_result, dataset, settings_obj

TRACE_SUMMARY_NAME = "traces-summary.json"
THRESHOLDS_PATH = Path("evaluation/thresholds.json")
RETRIEVAL_THRESHOLDS_PATH = Path("evaluation/thresholds-retrieval.json")
PUBLISHED = Path("evaluation/reports/2026-09-answers-baseline-v1")


def thresholds():
    """Return the bounds an answer-path report is held to."""
    return json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))


def retrieval_thresholds():
    """Return the bounds a retrieval-only report is held to."""
    return json.loads(RETRIEVAL_THRESHOLDS_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def a_report_directory(tmp_path, dataset, settings_obj):
    """Return a checked-in answer-path report directory, with its results written."""
    results = [an_answer_result()]
    manifest = a_manifest(dataset, settings_obj, results, run_records=[])
    return report.write_report(tmp_path / "answers-v1", manifest, results)


@pytest.fixture
def recorded(tmp_path):
    """Return an export file holding two recorded reader requests."""
    from tests.test_evaluation_traces import a_trace, write

    return write(
        tmp_path / "answer-traces.jsonl",
        a_trace(total=1.0),
        a_trace(total=1.0, outcome="abstained"),
    )


def a_threshold_file(tmp_path, **bounds):
    """Return a threshold file naming these bounds."""
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(bounds), encoding="utf-8")
    return path


class TestRenderingWithoutRunningAnything:
    """A report can be re-read without a provider, a key, or an index."""

    def test_it_writes_the_digest_into_the_report_directory(self, a_report_directory):
        """The digest belongs beside the numbers it describes."""
        cli.main(["--render", str(a_report_directory)])

        written = json.loads((a_report_directory / "digest.json").read_text())
        assert (a_report_directory / "digest.md").exists()
        assert written["provenance"]["dataset"] == "2026-09-labeled-cases-v1"

    def test_it_leaves_the_numbers_it_read_untouched(self, a_report_directory):
        """Re-reading a report must not quietly restate what it found."""
        before = (a_report_directory / "results" / "baseline.json").read_text()

        cli.main(["--render", str(a_report_directory)])

        assert (a_report_directory / "results" / "baseline.json").read_text() == before

    def test_it_reads_the_traces_named_and_stores_the_summary(
        self, a_report_directory, recorded
    ):
        """The raw traces are an operator's file; the summary is the report's."""
        cli.main(["--render", str(a_report_directory), "--traces", str(recorded)])

        stored = json.loads((a_report_directory / TRACE_SUMMARY_NAME).read_text())
        assert stored["traces"] == 2
        assert stored["outcomes"] == {"answered": 1, "abstained": 1}
        assert (
            json.loads((a_report_directory / "digest.json").read_text())["performance"][
                "traces"
            ]["traces"]
            == 2
        )

    def test_a_second_render_reproduces_the_first_from_what_was_stored(
        self, a_report_directory, recorded
    ):
        """A digest that only holds on the machine that made it is not a report."""
        cli.main(["--render", str(a_report_directory), "--traces", str(recorded)])
        first = (a_report_directory / "digest.json").read_text()

        cli.main(["--render", str(a_report_directory)])

        assert (a_report_directory / "digest.json").read_text() == first

    def test_a_report_with_no_traces_says_so_in_the_digest(self, a_report_directory):
        """Half a report is still a report, as long as it names the half it has."""
        cli.main(["--render", str(a_report_directory)])

        assert "no recorded traces" in (a_report_directory / "digest.md").read_text()

    def test_it_renders_a_retrieval_only_report_too(
        self, tmp_path, dataset, settings_obj
    ):
        """A report with no answers still has retrieval quality, and no cost."""
        results = [a_result()]
        manifest = a_manifest(dataset, settings_obj, results, run_records=[])
        directory = report.write_report(tmp_path / "retrieval-v1", manifest, results)

        cli.main(["--render", str(directory)])

        assert (directory / "digest.md").exists()

    def test_it_says_where_it_wrote_the_digest(self, a_report_directory, capsys):
        """The path is the one thing a person has to copy from the output."""
        cli.main(["--render", str(a_report_directory)])

        assert capsys.readouterr().out == f"Digest written to {a_report_directory}\n"


class TestTheThresholdGate:
    """CI fails on a crossed bound and on nothing else."""

    def test_it_passes_a_run_inside_every_declared_bound(
        self, a_report_directory, tmp_path
    ):
        """The common case has to be silence, or nobody reads the gate."""
        path = a_threshold_file(
            tmp_path,
            **{
                "quality.retrieval.hit_rate": {"min": 0.5},
                "cost.provider.usd": {"max": 1.0},
            },
        )

        cli.main(["--render", str(a_report_directory), "--check", str(path)])

    def test_it_fails_and_names_the_bound_a_run_crossed(
        self, a_report_directory, tmp_path, capsys
    ):
        """A gate that says which number moved is one somebody can act on."""
        path = a_threshold_file(
            tmp_path, **{"quality.retrieval.hit_rate": {"min": 0.95}}
        )

        with pytest.raises(SystemExit) as stopped:
            cli.main(["--render", str(a_report_directory), "--check", str(path)])

        assert stopped.value.code == 1
        out = capsys.readouterr().out
        assert "quality.retrieval.hit_rate" in out
        assert "0.9" in out

    def test_it_keeps_the_report_when_the_gate_fails(
        self, a_report_directory, tmp_path
    ):
        """The artifact is the reason to fail: it is what a person opens next."""
        path = a_threshold_file(
            tmp_path, **{"quality.retrieval.hit_rate": {"min": 0.95}}
        )

        with pytest.raises(SystemExit):
            cli.main(["--render", str(a_report_directory), "--check", str(path)])

        assert (a_report_directory / "digest.md").exists()
        assert (a_report_directory / "digest.json").exists()

    def test_a_bound_over_a_number_the_run_never_measured_fails_too(
        self, a_report_directory, tmp_path, capsys
    ):
        """A gate that passes what it cannot read is a gate nobody is watching."""
        path = a_threshold_file(
            tmp_path, **{"performance.traces.total_seconds.p95": {"max": 1.0}}
        )

        with pytest.raises(SystemExit) as stopped:
            cli.main(["--render", str(a_report_directory), "--check", str(path)])

        assert stopped.value.code == 1
        assert "not measured" in capsys.readouterr().out


def published_digest(directory=PUBLISHED):
    """Return the digest of a report this repository publishes."""
    return digest.digest(
        report.load_manifest(directory), list(report.load_results(directory).values())
    )


class TestTheCheckedInThresholds:
    """The bounds the gate reads are themselves part of the report."""

    def test_the_published_answer_report_meets_every_bound_it_declares(self):
        """A bound the published report fails is a regression or a mistake."""
        assert digest.regressions(published_digest(), thresholds()) == []

    def test_the_published_retrieval_reports_meet_their_own_bounds(self):
        """Retrieval runs are gated too, on the numbers they actually measure."""
        for name in (
            "2026-09-retrieval-baseline-v1",
            "2026-09-retrieval-tuning-v1",
        ):
            directory = Path("evaluation/reports") / name

            assert (
                digest.regressions(published_digest(directory), retrieval_thresholds())
                == []
            )

    def test_the_answer_thresholds_bound_quality_latency_and_cost(self):
        """A gate that only watches cost lets quality fall until somebody looks."""
        bounds = thresholds()

        assert any(name.startswith("quality.") for name in bounds)
        assert any(name.startswith("performance.") for name in bounds)
        assert any(name.startswith("cost.") for name in bounds)

    def test_the_retrieval_thresholds_bound_quality_and_latency(self):
        """A retrieval run has no tokens to bound, so it bounds what it measured."""
        bounds = retrieval_thresholds()

        assert any(name.startswith("quality.") for name in bounds)
        assert any(name.startswith("performance.") for name in bounds)
        assert not any(name.startswith("cost.") for name in bounds)
