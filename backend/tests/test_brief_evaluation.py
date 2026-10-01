"""
Brief tool-use evaluation: the task set, the graders, and the multi-trial run.

A brief is worth its extra calls only when the evidence answers the
question and the trajectory stayed inside its bounds. These tests pin the
deterministic half of that claim: a versioned task set over the six kinds of
brief question, graders for outcome and trajectory, consistency across
trials, and a report that compares direct retrieval against the brief and
the extended tools without requiring one exact tool path.
"""

import pytest

from evaluation import brief_eval
from evaluation.brief_eval import (
    BriefEvaluation,
    PaidNotAllowed,
    check_paid_allowance,
    configuration_of,
    estimate_cost_usd,
    grade_trials,
)
from evaluation.brief_grading import (
    BriefClaimRecord,
    BriefResultRecord,
    BriefTrajectory,
    EvidenceRecord,
    ToolCallRecord,
    grade_trajectory,
)
from evaluation.brief_tasks import (
    BriefTasksInvalid,
    load_brief_tasks,
    validate_payload,
)


def task_set():
    """Load the versioned brief task set."""
    return load_brief_tasks()


def evidence(evidence_id, document, content, page=3, method="hybrid"):
    """Return one held passage under a stable id."""
    return EvidenceRecord(
        evidence_id=evidence_id,
        document=document,
        content=content,
        page=page,
        method=method,
    )


def claim(text, supports=(), conflicts=(), status="supported"):
    """Return one brief claim."""
    return BriefClaimRecord(
        text=text, supports=tuple(supports), conflicts=tuple(conflicts), status=status
    )


def trajectory(task_id, trial=1, **overrides):
    """Return a successful two-document trajectory for the lookup task."""
    task = task_set().task("brief-cross-lookup-hit-rate")
    snippets = [item.snippet for item in task.expected_evidence]
    held = (
        evidence("E1", task.expected_evidence[0].document, f"holding {snippets[0]}"),
        evidence("E2", task.expected_evidence[1].document, f"holding {snippets[1]}"),
    )
    brief = BriefResultRecord(
        summary=task.expected_answer,
        claims=(claim(task.expected_answer, supports=("E1", "E2")),),
        gaps=(),
        abstained=False,
    )
    calls = (
        ToolCallRecord(
            tool="search_passages",
            arguments={"label": "A", "query": "hit rate"},
            evidence_ids=("E1",),
            latency_ms=12.0,
        ),
        ToolCallRecord(
            tool="search_passages",
            arguments={"label": "B", "query": "hit rate"},
            evidence_ids=("E2",),
            latency_ms=14.0,
        ),
        ToolCallRecord(
            tool="read_passages",
            arguments={"evidence_ids": ["E1", "E2"]},
            evidence_ids=("E1", "E2"),
            latency_ms=2.0,
        ),
    )
    base = {
        "task_id": task_id,
        "trial": trial,
        "configuration": "brief",
        "turns": 3,
        "tool_calls": calls,
        "outcome": "complete",
        "failure_category": "",
        "total_seconds": 4.0 + trial,
        "input_tokens": 1000,
        "output_tokens": 200,
        "cost_usd": 0.01,
        "evidence": held,
        "brief": brief,
        "invalid_citations": (),
    }
    base.update(overrides)
    return BriefTrajectory(**base)


class TestBriefTaskSet:
    """What the versioned task set promises."""

    def test_the_set_covers_all_six_kinds_of_brief_question(self):
        """Cross-document lookup, agreement, disagreement, missing evidence, unanswerable, and page citation each have a task."""
        counts = task_set().counts_by_category()

        assert set(counts) >= {
            "cross_lookup",
            "agreement",
            "disagreement",
            "missing_evidence",
            "unanswerable",
            "page_citation",
        }

    def test_every_task_reads_exactly_two_documents(self):
        """A brief is scoped to a pair, so a task that names anything else is refused."""
        for task in task_set().tasks:
            assert len(task.documents) == 2

    def test_an_unanswerable_task_expects_an_abstention(self):
        """A question neither document answers has one honest ending."""
        unanswerable = [
            task for task in task_set().tasks if task.category == "unanswerable"
        ]

        assert unanswerable
        for task in unanswerable:
            assert task.expects.expects_abstention
            assert task.expected_evidence == ()
            assert task.absent_terms

    def test_a_set_missing_a_category_is_refused(self):
        """A task set that cannot ask one kind of question is not balanced."""
        payload = {
            "version": "v",
            "reviewed_on": "2026-10-01",
            "documents": [
                {
                    "filename": "a.pdf",
                    "version": "1",
                    "sha256": "x",
                    "origin": "o",
                    "license": "l",
                },
                {
                    "filename": "b.pdf",
                    "version": "1",
                    "sha256": "y",
                    "origin": "o",
                    "license": "l",
                },
            ],
            "tasks": [],
        }

        with pytest.raises(BriefTasksInvalid):
            validate_payload(payload)


class TestOutcomeGrading:
    """What the brief says, graded without asking a model."""

    def test_coverage_is_a_set_comparison_not_a_path(self):
        """Two different orders of the same searches grade the same."""
        task = task_set().task("brief-cross-lookup-hit-rate")
        first = trajectory(task.id)
        swapped = trajectory(
            task.id,
            evidence=tuple(reversed(first.evidence)),
            tool_calls=tuple(reversed(first.tool_calls)),
        )

        assert grade_trajectory(task, first)["evidence_coverage"].outcome == "passed"
        assert grade_trajectory(task, swapped)["evidence_coverage"].outcome == "passed"

    def test_missing_passage_fails_coverage(self):
        """A brief that never held one side has not answered across documents."""
        task = task_set().task("brief-cross-lookup-hit-rate")
        partial = trajectory(task.id, evidence=trajectory(task.id).evidence[:1])

        assert grade_trajectory(task, partial)["evidence_coverage"].outcome == "failed"

    def test_disagreement_must_stay_contested(self):
        """Averaging two measures that pull apart into one winner fails."""
        task = task_set().task("brief-disagreement-single-representation")
        snippets = [item.snippet for item in task.expected_evidence]
        held = tuple(
            evidence(f"E{i + 1}", item.document, f"holding {item.snippet}")
            for i, item in enumerate(task.expected_evidence)
        )
        averaged = BriefTrajectory(
            task_id=task.id,
            trial=1,
            configuration="extended",
            turns=3,
            tool_calls=(
                ToolCallRecord(
                    tool="search_passages",
                    arguments={"label": "A", "query": "dense sparse"},
                ),
                ToolCallRecord(
                    tool="compare_evidence", arguments={"evidence_ids": ["E1", "E2"]}
                ),
            ),
            outcome="complete",
            total_seconds=3.0,
            input_tokens=500,
            output_tokens=100,
            cost_usd=0.005,
            evidence=held,
            brief=BriefResultRecord(
                summary="Sparse wins outright.",
                claims=(claim("Sparse wins outright.", supports=("E1", "E2")),),
            ),
        )

        assert grade_trajectory(task, averaged)["conflict_handling"].outcome == "failed"

    def test_unanswerable_must_abstain(self):
        """Answering from outside the pair is the failure the task exists to catch."""
        task = task_set().task("brief-unanswerable-hardware-cost")
        answered = BriefTrajectory(
            task_id=task.id,
            trial=1,
            configuration="brief",
            turns=2,
            tool_calls=(
                ToolCallRecord(
                    tool="search_passages", arguments={"label": "A", "query": "cost"}
                ),
            ),
            outcome="complete",
            total_seconds=2.0,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
            evidence=(),
            brief=BriefResultRecord(summary="It cost $10.", claims=()),
        )

        assert grade_trajectory(task, answered)["abstention"].outcome == "failed"

    def test_page_citation_needs_a_page(self):
        """A citation the reader cannot open is not a citation."""
        task = task_set().task("brief-page-citation-rerank-table")
        base = trajectory(task.id)
        pageless = BriefTrajectory(
            task_id=task.id,
            trial=1,
            configuration="extended",
            turns=base.turns,
            tool_calls=base.tool_calls,
            outcome="complete",
            total_seconds=3.0,
            input_tokens=500,
            output_tokens=100,
            cost_usd=0.005,
            evidence=tuple(
                evidence(item.evidence_id, item.document, item.content, page=None)
                for item in base.evidence
            ),
            brief=base.brief,
        )

        assert grade_trajectory(task, pageless)["page_citation"].outcome == "failed"

    def test_right_text_on_the_wrong_page_fails(self):
        """A citation naming the right passage on the wrong page fails."""
        task = task_set().task("brief-page-citation-rerank-table")
        held = tuple(
            evidence(f"E{position}", item.document, f"holding {item.snippet}", page=99)
            for position, item in enumerate(task.expected_evidence, start=1)
        )
        ids = [item.evidence_id for item in held]
        relocated = BriefTrajectory(
            task_id=task.id,
            trial=1,
            configuration="extended",
            turns=3,
            tool_calls=(
                ToolCallRecord(
                    tool="search_passages",
                    arguments={"label": "A", "query": "rerank"},
                    evidence_ids=(ids[0],),
                ),
                ToolCallRecord(
                    tool="read_page",
                    arguments={"label": "A", "page": 99},
                    evidence_ids=(ids[0],),
                ),
            ),
            outcome="complete",
            total_seconds=3.0,
            input_tokens=500,
            output_tokens=100,
            cost_usd=0.005,
            evidence=held,
            brief=BriefResultRecord(
                summary=task.expected_answer,
                claims=(claim(task.expected_answer, supports=tuple(ids)),),
            ),
        )

        assert grade_trajectory(task, relocated)["page_citation"].outcome == "failed"

    def test_a_ran_call_with_bad_arguments_fails_validity(self):
        """A malformed call the loop failed to refuse still fails validity."""
        task = task_set().task("brief-cross-lookup-hit-rate")
        base = trajectory(task.id)
        bad = trajectory(
            task.id,
            tool_calls=base.tool_calls
            + (
                ToolCallRecord(
                    tool="search_passages",
                    arguments={"label": "A"},
                ),
            ),
        )

        assert grade_trajectory(task, bad)["tool_validity"].outcome == "failed"


class TestTrajectoryGrading:
    """How the model behaved getting there."""

    def test_invalid_and_out_of_scope_calls_fail(self):
        """A trajectory that reached outside the pair or miscalled a tool fails."""
        task = task_set().task("brief-cross-lookup-hit-rate")
        base = trajectory(task.id)
        bad = trajectory(
            task.id,
            tool_calls=base.tool_calls
            + (
                ToolCallRecord(
                    tool="search_passages",
                    arguments={"label": "C", "query": "elsewhere"},
                    refusal="out_of_scope",
                ),
                ToolCallRecord(tool="browse", arguments={}, refusal="unknown_tool"),
            ),
        )
        grades = grade_trajectory(task, bad)

        assert grades["scope_enforcement"].outcome == "failed"
        assert grades["tool_validity"].outcome == "failed"

    def test_repeated_and_wasteful_calls_fail(self):
        """A model stuck re-asking or searching past the second query per side fails."""
        task = task_set().task("brief-cross-lookup-hit-rate")
        base = trajectory(task.id)
        repeated = trajectory(
            task.id,
            tool_calls=base.tool_calls
            + (
                ToolCallRecord(
                    tool="search_passages",
                    arguments={"label": "A", "query": "hit rate"},
                    refusal="repeated_call",
                ),
            ),
        )

        assert grade_trajectory(task, repeated)["repeated_calls"].outcome == "failed"

        wasteful = trajectory(
            task.id,
            tool_calls=tuple(
                ToolCallRecord(
                    tool="search_passages",
                    arguments={"label": "A", "query": f"q{i}"},
                )
                for i in range(9)
            ),
        )

        assert grade_trajectory(task, wasteful)["unnecessary_calls"].outcome == "failed"


class TestMultiTrialRun:
    """Consistency, spend, and baselines across trials."""

    def test_several_trials_report_consistency_latency_and_cost(self):
        """Three trials of one task report agreement, p50/p95, tokens, and cost."""
        task = task_set().task("brief-cross-lookup-hit-rate")
        evaluation = grade_trials(
            task_set(), [trajectory(task.id, trial=i) for i in (1, 2, 3)]
        )

        summary = next(item for item in evaluation.tasks if item.task_id == task.id)
        assert summary.trials == 3
        assert summary.outcome_consistency == 1.0
        assert summary.latency_p50 is not None
        assert summary.latency_p95 is not None
        assert evaluation.totals["tokens_total"] > 0
        assert evaluation.totals["cost_usd"] > 0

    def test_configurations_are_read_from_the_calls(self):
        """Direct, brief, and extended are what the trajectory used, not its label."""
        task = task_set().task("brief-cross-lookup-hit-rate")
        direct = trajectory(
            task.id, trial=1, tool_calls=(), evidence=(), configuration="direct"
        )
        extended_calls = trajectory(task.id, trial=2).tool_calls + (
            ToolCallRecord(
                tool="read_page",
                arguments={"label": "A", "page": 1},
                evidence_ids=("E3",),
            ),
        )
        extended = trajectory(
            task.id, trial=2, tool_calls=extended_calls, configuration="brief"
        )

        assert configuration_of(direct) == "direct"
        assert configuration_of(trajectory(task.id)) == "brief"
        assert configuration_of(extended) == "extended"

    def test_timeouts_and_failures_are_counted_not_averaged_away(self):
        """A timed-out trial is a timeout rate and a failure category, not a zero."""
        task = task_set().task("brief-cross-lookup-hit-rate")
        timed_out = trajectory(
            task.id, trial=9, outcome="timeout", failure_category="timeout"
        )
        evaluation = grade_trials(task_set(), [trajectory(task.id), timed_out])

        assert evaluation.totals["timeout_rate"] == 0.5
        assert evaluation.totals["failures"] == {"timeout": 1}


class TestPaidSchedule:
    """Deterministic checks run routinely; paid trials need a schedule and a limit."""

    def test_paid_trials_need_an_explicit_schedule(self):
        """Recording new trajectories without opting in is refused."""
        with pytest.raises(PaidNotAllowed):
            check_paid_allowance(allow_paid=False, estimated_cost_usd=0.1)

    def test_paid_trials_refuse_past_the_cost_limit(self):
        """A run that would spend past the limit stops before the first call."""
        tasks = len(task_set().tasks)
        estimate = estimate_cost_usd(tasks, trials=3, per_trial_usd=10.0)

        with pytest.raises(PaidNotAllowed):
            check_paid_allowance(
                allow_paid=True, estimated_cost_usd=estimate, limit_usd=5.0
            )

    def test_evaluation_is_a_paid_free_record(self):
        """The deterministic evaluation carries no paid verdicts."""
        evaluation = grade_trials(
            task_set(), [trajectory("brief-cross-lookup-hit-rate")]
        )

        assert isinstance(evaluation, BriefEvaluation)
        assert evaluation.totals["trials"] == 1
