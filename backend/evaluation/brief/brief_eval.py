"""
Running the Research Brief task set without paying a provider, and measuring it.

A brief is nondeterministic: the same question asked twice can search in a
different order and still end with the same evidence. The runner therefore
grades recorded trajectories rather than live models, aggregates several
trials of each task, and reports consistency beside quality. Two different
orders of the same searches grade the same, because coverage is a set
comparison and validity is a property of the whole trajectory rather than of
one exact path.

Three configurations are compared side by side: direct retrieval with no
tools, the two-document brief with search and reading, and the extended
brief with page reading and evidence comparison. The deterministic checks
here run routinely with no key; a paid model trial that records new
trajectories runs only on a controlled schedule with a cost limit, and the
gate for that lives in this module rather than in the caller.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from evaluation.brief import brief_grading
from evaluation.brief.brief_grading import (
    BriefClaimRecord,
    BriefResultRecord,
    BriefTrajectory,
    EvidenceRecord,
    ToolCallRecord,
)
from evaluation.answers.metrics import latency_summary, percentile

#: The configurations a brief report compares, in the order it compares them.
#: Direct retrieval asks once with no tools; the brief searches and reads;
#: the extended brief may also read pages and compare what it holds.
DIRECT = "direct"
BRIEF = "brief"
EXTENDED = "extended"
CONFIGURATIONS = (DIRECT, BRIEF, EXTENDED)

#: Which tools each configuration may have used. A trajectory that used a page
#: read is an extended trajectory even when it was recorded as a brief, so
#: the configuration is read from the calls rather than trusted from the label.
EXTENDED_TOOLS = ("read_page", "compare_evidence")

#: How many paid trials one task is measured with. One run can be lucky;
#: three shows whether the brief finds the same evidence twice.
PAID_TRIALS_PER_TASK = 3

#: The most a paid model trial may estimate before it refuses to record. A
#: brief burns several model calls per question, so the bound is per task set
#: run rather than per call, and a run that would spend past it stops before
#: the first paid call rather than halfway through the set.
DEFAULT_PAID_COST_LIMIT_USD = 5.0

#: Why a paid trial did not run. Carried in the report so a reader can tell a
#: skipped schedule from a measured zero.
PAID_SKIPPED_REASON = (
    "paid model trials run on a controlled schedule with a cost limit; "
    "this report holds deterministic checks only"
)


class PaidNotAllowed(RuntimeError):
    """A paid brief trial was asked for outside its schedule or past its limit."""


def configuration_of(trajectory: BriefTrajectory) -> str:
    """Return the configuration a trajectory actually used, from its calls."""
    tools = {call.tool for call in trajectory.ran_calls}
    if tools & set(EXTENDED_TOOLS):
        return EXTENDED
    if tools:
        return BRIEF
    return DIRECT


@dataclass
class TrialResult:
    """One graded trial: the trajectory, its grades, and what it cost."""

    task_id: str
    trial: int
    configuration: str
    outcome: str
    turns: int
    tool_calls: int
    total_seconds: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    grades: dict[str, dict[str, Any]] = field(default_factory=dict)
    failure_category: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the trial as a report stores it."""
        return asdict(self)


@dataclass
class TaskSummary:
    """One task measured over several trials, with its consistency."""

    task_id: str
    category: str
    trials: int
    configurations: list[str]
    outcome_consistency: float | None
    grade_pass_rate: dict[str, float]
    latency_p50: float | None
    latency_p95: float | None
    turns_p50: float | None
    turns_p95: float | None
    tool_calls_p50: float | None
    tool_calls_p95: float | None
    tokens_p50: float | None
    tokens_p95: float | None
    cost_usd: float
    timeout_rate: float
    failures: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Return the task summary as a report stores it."""
        return asdict(self)


@dataclass
class BriefEvaluation:
    """Every trial, every task summary, and the configuration comparison."""

    task_set: str
    trials: list[TrialResult]
    tasks: list[TaskSummary]
    configurations: list[dict[str, Any]]
    totals: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the evaluation as a report stores it."""
        return {
            "task_set": self.task_set,
            "trials": [trial.to_dict() for trial in self.trials],
            "tasks": [task.to_dict() for task in self.tasks],
            "configurations": self.configurations,
            "totals": self.totals,
        }


def check_paid_allowance(
    *,
    allow_paid: bool,
    estimated_cost_usd: float,
    limit_usd: float = DEFAULT_PAID_COST_LIMIT_USD,
) -> None:
    """Refuse a paid brief trial outside its schedule or past its limit."""
    if not allow_paid:
        raise PaidNotAllowed(
            "paid brief trials run on a controlled schedule; "
            "pass --allow-paid to record new trajectories"
        )
    if estimated_cost_usd > limit_usd:
        raise PaidNotAllowed(
            f"estimated ${estimated_cost_usd:.2f} is past the "
            f"${limit_usd:.2f} limit for one paid brief run"
        )


def estimate_cost_usd(
    tasks: int, trials: int = PAID_TRIALS_PER_TASK, per_trial_usd: float = 0.05
) -> float:
    """Return a rough paid-run estimate so the gate can refuse before spending."""
    return round(tasks * trials * per_trial_usd, 4)


def grade_trials(task_set: Any, trajectories: list[BriefTrajectory]) -> BriefEvaluation:
    """Grade every trajectory and summarize each task over its trials."""
    by_task: dict[str, Any] = {task.id: task for task in task_set.tasks}
    trials: list[TrialResult] = []
    for trajectory in trajectories:
        task = by_task.get(trajectory.task_id)
        if task is None:
            raise KeyError(f"{trajectory.task_id} is not a task in {task_set.version}")
        grades = brief_grading.grade_trajectory(task, trajectory)
        trials.append(
            TrialResult(
                task_id=trajectory.task_id,
                trial=trajectory.trial,
                configuration=configuration_of(trajectory),
                outcome=trajectory.outcome,
                turns=trajectory.turns,
                tool_calls=len(trajectory.ran_calls),
                total_seconds=trajectory.total_seconds,
                input_tokens=trajectory.input_tokens,
                output_tokens=trajectory.output_tokens,
                cost_usd=trajectory.cost_usd,
                grades={name: grade.to_dict() for name, grade in grades.items()},
                failure_category=trajectory.failure_category,
            )
        )
    tasks = [_summarize_task(task_set, task_id, trials) for task_id in by_task]
    return BriefEvaluation(
        task_set=task_set.version,
        trials=trials,
        tasks=[task for task in tasks if task is not None],
        configurations=_compare_configurations(trials),
        totals=_totals(trials),
    )


def _summarize_task(
    task_set: Any, task_id: str, trials: list[TrialResult]
) -> TaskSummary | None:
    """Summarize one task's trials: quality, consistency, latency, and cost."""
    task_trials = [trial for trial in trials if trial.task_id == task_id]
    if not task_trials:
        return None
    category = next(task.category for task in task_set.tasks if task.id == task_id)
    outcomes = [trial.outcome for trial in task_trials]
    majority = Counter(outcomes).most_common(1)[0][1]
    consistency = majority / len(task_trials) if len(task_trials) > 1 else 1.0
    grade_names = sorted(task_trials[0].grades)
    pass_rate = {}
    for name in grade_names:
        graded = [
            trial for trial in task_trials if trial.grades[name]["outcome"] != "unknown"
        ]
        if not graded:
            continue
        pass_rate[name] = round(
            sum(1 for trial in graded if trial.grades[name]["outcome"] == "passed")
            / len(graded),
            3,
        )
    latencies = [trial.total_seconds for trial in task_trials]
    turns = [float(trial.turns) for trial in task_trials]
    calls = [float(trial.tool_calls) for trial in task_trials]
    tokens = [float(trial.input_tokens + trial.output_tokens) for trial in task_trials]
    failures: dict[str, int] = {}
    for trial in task_trials:
        if trial.outcome not in ("complete", "abstained", "partial"):
            key = trial.failure_category or trial.outcome
            failures[key] = failures.get(key, 0) + 1
    timeouts = sum(1 for trial in task_trials if trial.outcome == "timeout")
    return TaskSummary(
        task_id=task_id,
        category=category,
        trials=len(task_trials),
        configurations=sorted({trial.configuration for trial in task_trials}),
        outcome_consistency=round(consistency, 3),
        grade_pass_rate={name: round(rate, 3) for name, rate in pass_rate.items()},
        latency_p50=percentile(latencies, 0.5),
        latency_p95=percentile(latencies, 0.95),
        turns_p50=percentile(turns, 0.5),
        turns_p95=percentile(turns, 0.95),
        tool_calls_p50=percentile(calls, 0.5),
        tool_calls_p95=percentile(calls, 0.95),
        tokens_p50=percentile(tokens, 0.5),
        tokens_p95=percentile(tokens, 0.95),
        cost_usd=round(sum(trial.cost_usd for trial in task_trials), 6),
        timeout_rate=round(timeouts / len(task_trials), 3),
        failures=failures,
    )


def _totals(trials: list[TrialResult]) -> dict[str, Any]:
    """Summarize the whole run: latency tails, spend, timeouts, and failures."""
    latencies = [trial.total_seconds for trial in trials]
    turns = [float(trial.turns) for trial in trials]
    calls = [float(trial.tool_calls) for trial in trials]
    tokens = [trial.input_tokens + trial.output_tokens for trial in trials]
    failures: dict[str, int] = {}
    for trial in trials:
        if trial.outcome not in ("complete", "abstained", "partial"):
            key = trial.failure_category or trial.outcome
            failures[key] = failures.get(key, 0) + 1
    latency = latency_summary(latencies)
    return {
        "trials": len(trials),
        "latency_p50": latency.p50,
        "latency_p95": latency.p95,
        "turns_p50": percentile(turns, 0.5),
        "turns_p95": percentile(turns, 0.95),
        "tool_calls_p50": percentile(calls, 0.5),
        "tool_calls_p95": percentile(calls, 0.95),
        "tokens_total": sum(tokens),
        "tokens_p50": percentile([float(value) for value in tokens], 0.5),
        "tokens_p95": percentile([float(value) for value in tokens], 0.95),
        "cost_usd": round(sum(trial.cost_usd for trial in trials), 6),
        "timeout_rate": round(
            sum(1 for trial in trials if trial.outcome == "timeout") / len(trials),
            3,
        )
        if trials
        else 0.0,
        "failures": failures,
    }


def _pass_rate(trials: list[TrialResult], grade: str) -> float | None:
    """
    Return the share of graded trials where one grader passed, or None for none.

    Unknown is not a failure: it is a task the grader could not decide, so it
    is left out of the mean rather than scored as a zero.
    """
    graded = [trial for trial in trials if trial.grades[grade]["outcome"] != "unknown"]
    if not graded:
        return None
    return round(
        sum(1 for trial in graded if trial.grades[grade]["outcome"] == "passed")
        / len(graded),
        3,
    )


def _compare_configurations(trials: list[TrialResult]) -> list[dict[str, Any]]:
    """Compare the three configurations on the same tasks, as pass rates."""
    rows = []
    for configuration in CONFIGURATIONS:
        subset = [trial for trial in trials if trial.configuration == configuration]
        latencies = [trial.total_seconds for trial in subset]
        rows.append(
            {
                "configuration": configuration,
                "trials": len(subset),
                "evidence_coverage": _pass_rate(subset, "evidence_coverage"),
                "correctness": _pass_rate(subset, "correctness"),
                "citation_precision": _pass_rate(subset, "citation_precision"),
                "conflict_handling": _pass_rate(subset, "conflict_handling"),
                "abstention": _pass_rate(subset, "abstention"),
                "completion": _pass_rate(subset, "completion"),
                "latency_p50": percentile(latencies, 0.5),
                "latency_p95": percentile(latencies, 0.95),
                "cost_usd": round(sum(trial.cost_usd for trial in subset), 6),
            }
        )
    return rows


def demo_trajectories(task_set: Any) -> list[BriefTrajectory]:
    """
    Build deterministic trajectories for every task, without calling a model.

    Trials 1 and 2 of each task are the trajectories a good brief would
    record: all expected evidence held, valid calls only, the outcome the
    task allows. Trial 2 repeats the same evidence in a different tool
    order, which is what proves coverage is a set comparison rather than
    one exact path. Trial 3 repeats the first with different timings, so
    consistency is measured over three trials per task rather than one
    lucky run. A direct-retrieval trial with no tool calls is added for the
    lookup tasks, so the report compares all three configurations on the
    same questions. One partial, one conflicting, and one failed trajectory
    are included so the published report shows each kind.
    """
    trajectories: list[BriefTrajectory] = []
    trial = 0
    for task in task_set.tasks:
        trial += 1
        trajectories.append(_good_trajectory(task, trial=trial, seconds=3.0 + trial))
        trial += 1
        trajectories.append(
            _good_trajectory(task, trial=trial, seconds=4.0 + trial, reverse=True)
        )
        trial += 1
        trajectories.append(
            _good_trajectory(task, trial=trial, seconds=5.0 + trial * 0.5)
        )
    trajectories.append(_direct_trajectory(task_set, trial=trial + 1))
    trajectories.append(_partial_trajectory(task_set, trial=trial + 2))
    trajectories.append(_failed_trajectory(task_set, trial=trial + 3))
    return trajectories


def _held_evidence(task: Any, page: int | None = 3) -> tuple[EvidenceRecord, ...]:
    """
    Return one held passage per expected snippet, under stable ids.

    Where the task declares the page a passage sits on, the held evidence
    carries that page, so page-citation grading checks the right text on
    the right page rather than any page.
    """
    held: list[EvidenceRecord] = []
    for position, item in enumerate(task.expected_evidence, start=1):
        held.append(
            EvidenceRecord(
                evidence_id=f"E{position}",
                document=item.document,
                content=f"holding {item.snippet} with surrounding passage text",
                page=getattr(item, "page", None) or page,
                method="hybrid",
            )
        )
    return tuple(held)


def _search_calls(task: Any, reverse: bool = False) -> tuple[ToolCallRecord, ...]:
    """Return one search per document plus reads, optionally reversed."""
    labels = ["A", "B"]
    if reverse:
        labels = list(reversed(labels))
    calls = [
        ToolCallRecord(
            tool="search_passages",
            arguments={"label": label, "query": task.question[:80]},
            evidence_ids=(f"E{position}",),
            latency_ms=10.0 + position,
        )
        for position, label in enumerate(labels, start=1)
    ]
    if task.expected_evidence:
        calls.append(
            ToolCallRecord(
                tool="read_passages",
                arguments={
                    "evidence_ids": [
                        f"E{i + 1}" for i in range(len(task.expected_evidence))
                    ]
                },
                evidence_ids=tuple(
                    f"E{i + 1}" for i in range(len(task.expected_evidence))
                ),
                latency_ms=2.0,
            )
        )
    if task.expects.needs_compare and len(task.expected_evidence) >= 2:
        calls.append(
            ToolCallRecord(
                tool="compare_evidence",
                arguments={"evidence_ids": ["E1", "E2"]},
                evidence_ids=("E1", "E2"),
                latency_ms=3.0,
            )
        )
    if task.expects.needs_page_citation:
        calls.append(
            ToolCallRecord(
                tool="read_page",
                arguments={"label": "A", "page": 1},
                evidence_ids=("E1",),
                latency_ms=5.0,
            )
        )
    return tuple(calls)


def _good_brief(task: Any, held: tuple[EvidenceRecord, ...]) -> BriefResultRecord:
    """Return the final brief a good run would write for one task."""
    ids = [item.evidence_id for item in held]
    if task.expects.expects_abstention:
        return BriefResultRecord(summary="", claims=(), gaps=(), abstained=True)
    if task.expects.expects_conflict:
        return BriefResultRecord(
            summary=task.expected_answer,
            claims=(
                BriefClaimRecord(
                    text=task.expected_answer,
                    supports=tuple(ids[:1]),
                    conflicts=tuple(ids[1:2]) if len(ids) > 1 else (),
                    status="contested",
                ),
            ),
            gaps=(),
            abstained=False,
        )
    if task.expects.expects_gap:
        return BriefResultRecord(
            summary=task.expected_answer,
            claims=(
                BriefClaimRecord(
                    text=task.expected_answer,
                    supports=tuple(ids),
                    conflicts=(),
                    status="supported" if ids else "unresolved",
                ),
            ),
            gaps=("The pair names no embedding model, so that half is unanswered.",)
            if task.category == "missing_evidence"
            else (),
            abstained=False,
        )
    return BriefResultRecord(
        summary=task.expected_answer,
        claims=(
            BriefClaimRecord(
                text=task.expected_answer,
                supports=tuple(ids),
                conflicts=(),
                status="supported",
            ),
        )
        if ids
        else (),
        gaps=(),
        abstained=False,
    )


def _good_trajectory(
    task: Any, *, trial: int, seconds: float, reverse: bool = False
) -> BriefTrajectory:
    """Return a good deterministic trajectory for one task."""
    held = _held_evidence(task)
    outcome = "complete"
    if task.expects.expects_abstention:
        held = ()
        outcome = "abstained"
    elif task.expects.expects_gap:
        outcome = "partial"
    return BriefTrajectory(
        task_id=task.id,
        trial=trial,
        configuration="brief",
        turns=3,
        tool_calls=_search_calls(task, reverse=reverse),
        outcome=outcome,
        total_seconds=seconds,
        input_tokens=900 + trial * 10,
        output_tokens=180 + trial * 5,
        cost_usd=0.008,
        evidence=held,
        brief=_good_brief(task, held),
    )


def _direct_trajectory(task_set: Any, *, trial: int) -> BriefTrajectory:
    """Return a direct-retrieval baseline: one lookup task, no tool calls."""
    task = next(task for task in task_set.tasks if task.category == "cross_lookup")
    return BriefTrajectory(
        task_id=task.id,
        trial=trial,
        configuration="direct",
        turns=1,
        tool_calls=(),
        outcome="complete",
        total_seconds=1.2,
        input_tokens=400,
        output_tokens=120,
        cost_usd=0.002,
        evidence=(),
        brief=BriefResultRecord(
            summary=task.expected_answer,
            claims=(
                BriefClaimRecord(
                    text=task.expected_answer, supports=(), status="unresolved"
                ),
            ),
        ),
    )


def _partial_trajectory(task_set: Any, *, trial: int) -> BriefTrajectory:
    """Return a partial run: a missing-evidence task stopped with a gap."""
    task = next(task for task in task_set.tasks if task.category == "missing_evidence")
    held = _held_evidence(task)[:1]
    return BriefTrajectory(
        task_id=task.id,
        trial=trial,
        configuration="brief",
        turns=6,
        tool_calls=_search_calls(task),
        outcome="partial",
        failure_category="",
        total_seconds=9.5,
        input_tokens=1100,
        output_tokens=220,
        cost_usd=0.012,
        evidence=held,
        brief=BriefResultRecord(
            summary=task.expected_answer,
            claims=(
                BriefClaimRecord(
                    text=task.expected_answer,
                    supports=tuple(item.evidence_id for item in held),
                    status="supported",
                ),
            ),
            gaps=("The pair names no embedding model, so that half is unanswered.",),
        ),
    )


def _failed_trajectory(task_set: Any, *, trial: int) -> BriefTrajectory:
    """Return a failed run: a lookup task whose model call timed out."""
    task = next(task for task in task_set.tasks if task.category == "cross_lookup")
    return BriefTrajectory(
        task_id=task.id,
        trial=trial,
        configuration="brief",
        turns=2,
        tool_calls=(
            ToolCallRecord(
                tool="search_passages",
                arguments={"label": "A", "query": task.question[:80]},
                evidence_ids=("E1",),
                latency_ms=11.0,
            ),
        ),
        outcome="timeout",
        failure_category="timeout",
        total_seconds=30.0,
        input_tokens=900,
        output_tokens=0,
        cost_usd=0.004,
        evidence=_held_evidence(task)[:1],
        brief=BriefResultRecord(summary="", claims=(), gaps=(), abstained=False),
    )


def pick_representatives(
    evaluation: BriefEvaluation, trajectories: list[BriefTrajectory]
) -> dict[str, dict[str, Any]]:
    """Pick one successful, partial, conflicting, and failed trajectory."""
    by_key = {(trial.task_id, trial.trial): trial for trial in evaluation.trials}
    held = {(item.task_id, item.trial): item for item in trajectories}

    def record(key: tuple[str, int]) -> dict[str, Any]:
        trial = by_key[key]
        trajectory = held[key]
        return {
            "task_id": trial.task_id,
            "trial": trial.trial,
            "configuration": trial.configuration,
            "outcome": trial.outcome,
            "turns": trial.turns,
            "tool_calls": trial.tool_calls,
            "total_seconds": trial.total_seconds,
            "cost_usd": trial.cost_usd,
            "grades": trial.grades,
            "brief": {
                "summary": trajectory.brief.summary[:500],
                "claims": [
                    {
                        "text": claim.text[:200],
                        "supports": list(claim.supports),
                        "conflicts": list(claim.conflicts),
                        "status": claim.status,
                    }
                    for claim in trajectory.brief.claims
                ],
                "gaps": list(trajectory.brief.gaps),
                "abstained": trajectory.brief.abstained,
            },
            "tools": [
                {
                    "tool": call.tool,
                    "refusal": call.refusal,
                    "evidence_ids": list(call.evidence_ids),
                    "latency_ms": call.latency_ms,
                }
                for call in trajectory.tool_calls
            ],
        }

    successful = partial = conflicting = failed = None
    for key, trial in by_key.items():
        trajectory = held[key]
        failed_grades = [
            name for name, grade in trial.grades.items() if grade["outcome"] == "failed"
        ]
        if successful is None and trial.outcome == "complete" and not failed_grades:
            successful = key
        if partial is None and trial.outcome == "partial":
            partial = key
        if conflicting is None and any(
            claim.status == "contested" for claim in trajectory.brief.claims
        ):
            conflicting = key
        if failed is None and trial.outcome in ("failed", "timeout"):
            failed = key
    picked: dict[str, dict[str, Any]] = {}
    if successful is not None:
        picked["successful"] = record(successful)
    if partial is not None:
        picked["partial"] = record(partial)
    if conflicting is not None:
        picked["conflicting"] = record(conflicting)
    if failed is not None:
        picked["failed"] = record(failed)
    return picked


def write_brief_report(
    directory: Path,
    *,
    task_set: Any,
    evaluation: BriefEvaluation,
    trajectories: list[BriefTrajectory],
    revision: dict[str, Any] | None = None,
) -> Path:
    """Write a brief tool-use report a reviewer can read without running anything."""
    from evaluation.reporting import report as report_module

    directory = Path(directory)
    results_dir = directory / report_module.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = evaluation.to_dict()
    payload["representative_trajectories"] = pick_representatives(
        evaluation, trajectories
    )
    payload["paid_trials"] = {
        "measured": False,
        "reason": PAID_SKIPPED_REASON,
        "trials_per_task": PAID_TRIALS_PER_TASK,
        "cost_limit_usd": DEFAULT_PAID_COST_LIMIT_USD,
    }
    (results_dir / "brief-tool-use.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "report_version": report_module.REPORT_VERSION,
        "mode": "brief-tool-use",
        "revision": revision or report_module.git_revision(),
        "task_set": {
            "version": task_set.version,
            "reviewed_on": task_set.reviewed_on,
            "tasks": len(task_set.tasks),
            "categories": task_set.counts_by_category()
            if hasattr(task_set, "counts_by_category")
            else {},
        },
        "documents": list(task_set.documents),
        "configurations": list(CONFIGURATIONS),
        "totals": evaluation.totals,
    }
    (directory / report_module.MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (directory / report_module.SUMMARY_NAME).write_text(
        render_brief_report(
            task_set, evaluation, payload["representative_trajectories"]
        ),
        encoding="utf-8",
    )
    return directory


def render_brief_report(
    task_set: Any, evaluation: BriefEvaluation, representatives: dict[str, Any]
) -> str:
    """Return the brief tool-use report as the markdown a reviewer reads first."""
    totals = evaluation.totals
    lines = [
        f"# Research Brief tool use: task set {task_set.version}",
        "",
        f"{len(task_set.tasks)} tasks over "
        f"{evaluation.totals['trials']} deterministic trials. "
        "Each task is measured several times because a brief chooses its own "
        "tool order; consistency across trials is reported beside quality so "
        "one lucky run cannot pass the set.",
        "",
        "## What this report measured",
        "",
        "- **outcome**: evidence coverage, correctness, citation precision "
        "and recall, page citations, conflict handling, gaps, and abstention, "
        "graded deterministically from recorded trajectories",
        "- **trajectory**: tool-call validity, scope enforcement, repeated "
        "calls, unnecessary calls, and completion status for every trial",
        "- **equivalence**: coverage is a set comparison over held evidence, "
        "so two different orders of the same searches grade the same; no "
        "task requires one exact tool path",
        "- **paid model trials**: not measured here — they run on a "
        "controlled schedule with a cost limit "
        f"(${DEFAULT_PAID_COST_LIMIT_USD:.2f} per run, "
        f"{PAID_TRIALS_PER_TASK} trials per task)",
        "",
        "## Totals",
        "",
        f"Trials {totals['trials']}, latency "
        f"p50 {_seconds(totals['latency_p50'])} p95 {_seconds(totals['latency_p95'])}, "
        f"turns p50 {_number(totals['turns_p50'])} p95 "
        f"{_number(totals['turns_p95'])}, tool calls p50 "
        f"{_number(totals['tool_calls_p50'])} p95 "
        f"{_number(totals['tool_calls_p95'])}, tokens "
        f"{totals['tokens_total']} (p50 {_number(totals['tokens_p50'])} p95 "
        f"{_number(totals['tokens_p95'])}), "
        f"cost ${totals['cost_usd']:.4f}, timeout rate {totals['timeout_rate']:.0%}.",
        (
            "Failures: "
            + ", ".join(
                f"{name} {count}" for name, count in sorted(totals["failures"].items())
            )
            + "."
        )
        if totals["failures"]
        else "No trial failed or timed out.",
        "",
        "## Tasks",
        "",
        "| Task | Category | Trials | Consistency | Coverage | Correctness | "
        "Conflict | Abstention | p50 / p95 (s) | Cost ($) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for task in evaluation.tasks:
        rates = task.grade_pass_rate
        lines.append(
            f"| `{task.task_id}` | {task.category} | {task.trials} | "
            f"{task.outcome_consistency:.0%} | "
            f"{_rate(rates.get('evidence_coverage'))} | "
            f"{_rate(rates.get('correctness'))} | "
            f"{_rate(rates.get('conflict_handling'))} | "
            f"{_rate(rates.get('abstention'))} | "
            f"{_seconds(task.latency_p50)} / {_seconds(task.latency_p95)} | "
            f"{task.cost_usd:.4f} |"
        )
    lines += [
        "",
        "## Configurations",
        "",
        "Direct retrieval asks once with no tools; the brief searches and "
        "reads; the extended brief may also read pages and compare evidence.",
        "",
        "| Configuration | Trials | Coverage | Correctness | Citation | Conflict | "
        "Abstention | p50 / p95 (s) | Cost ($) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in evaluation.configurations:
        lines.append(
            f"| `{row['configuration']}` | {row['trials']} | "
            f"{_rate(row['evidence_coverage'])} | {_rate(row['correctness'])} | "
            f"{_rate(row['citation_precision'])} | {_rate(row['conflict_handling'])} | "
            f"{_rate(row['abstention'])} | "
            f"{_seconds(row['latency_p50'])} / {_seconds(row['latency_p95'])} | "
            f"{row['cost_usd']:.4f} |"
        )
    lines += ["", "## Representative trajectories", ""]
    for label in ("successful", "partial", "conflicting", "failed"):
        record = representatives.get(label)
        if record is None:
            lines.append(f"### {label.capitalize()}: none recorded")
            lines.append("")
            continue
        lines.append(
            f"### {label.capitalize()}: `{record['task_id']}` "
            f"trial {record['trial']} ({record['configuration']}, "
            f"{record['outcome']}, {record['turns']} turns, "
            f"{record['tool_calls']} tool calls, "
            f"{_seconds(record['total_seconds'])}, ${record['cost_usd']:.4f})"
        )
        lines.append("")
        tools = ", ".join(
            f"{entry['tool']}" + (f" [{entry['refusal']}]" if entry["refusal"] else "")
            for entry in record["tools"]
        )
        lines.append(f"Tools: {tools or 'no tool calls (direct retrieval)'}.")
        lines.append("")
        for claim in record["brief"]["claims"]:
            lines.append(
                f"- {claim['status']}: {claim['text']} "
                f"(supports {', '.join(claim['supports']) or '—'}; "
                f"conflicts {', '.join(claim['conflicts']) or '—'})"
            )
        for gap in record["brief"]["gaps"]:
            lines.append(f"- gap: {gap}")
        if record["brief"]["abstained"]:
            lines.append("- abstained rather than completing from model knowledge.")
        lines.append("")
    lines += [
        "## Reproducing this report",
        "",
        "```",
        "uv run pytest backend/tests/test_brief_evaluation.py",
        "uv run python -m evaluation.cli --brief-eval --brief-report <directory>",
        "```",
        "",
        "Deterministic checks run routinely with no key. Paid model trials "
        "that record new trajectories run on a controlled schedule: "
        "`--brief-eval --allow-paid` refuses when the estimate is past the "
        f"${DEFAULT_PAID_COST_LIMIT_USD:.2f} limit.",
        "",
    ]
    return "\n".join(lines)


def _seconds(value: Any) -> str:
    """Return seconds as a table cell, or a dash when there are none."""
    if value is None:
        return "-"
    return f"{float(value):.2f}s"


def _number(value: Any) -> str:
    """Return a percentile as a table cell, or a dash when there is none."""
    if value is None:
        return "-"
    return f"{float(value):.1f}"


def _rate(value: Any) -> str:
    """Return a pass rate as a table cell, or a dash when there is none."""
    if value is None:
        return "-"
    return f"{float(value):.0%}"


__all__ = [
    "BRIEF",
    "CONFIGURATIONS",
    "DIRECT",
    "EXTENDED",
    "BriefEvaluation",
    "TrialResult",
    "check_paid_allowance",
    "estimate_cost_usd",
    "grade_trials",
    "write_brief_report",
]
