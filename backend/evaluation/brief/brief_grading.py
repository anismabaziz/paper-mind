"""
Deterministic grading for Research Brief tool use.

A brief is worth its extra calls only when the evidence it ends with answers
the question, the citations point at that evidence, disagreements stay
disagreements, and the trajectory that got there stayed inside its bounds.
Every check here is computed from text the run already recorded, so the same
trajectory always gets the same grades and a disagreement can be traced to
the exact claim or call that caused it.

Two halves, deliberately separate. The outcome half asks what the brief says:
evidence coverage, correctness, citation precision and recall, page
citations, conflict handling, gaps, and abstention. The trajectory half asks
how the model behaved getting there: valid calls, scope discipline, repeats,
waste, and completion. A brief that quotes the right passages after ten
refused calls is a different finding from one that searched twice, and the
report keeps both.

No check requires one exact tool path. Coverage is a set comparison over the
held evidence, conflict is a property of the final claims, and validity is a
property of the whole trajectory, so two different orders of the same
searches grade the same.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evaluation.answers.graders import content_words
from evaluation.answers.metrics import contains_snippet

PASSED = "passed"
FAILED = "failed"
UNKNOWN = "unknown"

#: How much of the expected answer's wording the brief has to reuse before it
#: counts as saying the same thing. Half the content words is the point where
#: a brief that merely shares the topic stops looking like one that answered.
CORRECTNESS_OVERLAP = 0.5

#: The most tool calls that ran that a trajectory may hold before the waste
#: grader calls it wasteful. Briefs are bounded, and a run that searched
#: eight times for a two-document question was exploring rather than reading.
MAX_RAN_CALLS = 8

#: The most searches per document label before the waste grader calls the
#: extra ones unnecessary. Two searches per side is a refined query; the
#: third is the model asking the same shelf again.
MAX_SEARCHES_PER_LABEL = 2


@dataclass(frozen=True)
class ToolCallRecord:
    """One tool call the trajectory records, whether it ran or was refused."""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    refusal: str | None = None
    evidence_ids: tuple[str, ...] = ()
    latency_ms: float = 0.0

    @property
    def ran(self) -> bool:
        """Report whether the loop ran this call rather than refusing it."""
        return not self.refusal


@dataclass(frozen=True)
class BriefClaimRecord:
    """One claim of the final brief, with the evidence on each side of it."""

    text: str
    supports: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    status: str = "supported"


@dataclass(frozen=True)
class BriefResultRecord:
    """The final brief: what it said and whether it held its tongue."""

    summary: str = ""
    claims: tuple[BriefClaimRecord, ...] = ()
    gaps: tuple[str, ...] = ()
    abstained: bool = False


@dataclass(frozen=True)
class EvidenceRecord:
    """One passage the brief holds, as the trajectory records it."""

    evidence_id: str
    document: str
    content: str
    page: int | None = None
    method: str = "hybrid"


@dataclass(frozen=True)
class BriefTrajectory:
    """Everything one brief trial did, in the shape the graders read."""

    task_id: str
    trial: int
    configuration: str
    turns: int
    tool_calls: tuple[ToolCallRecord, ...] = ()
    outcome: str = "complete"
    failure_category: str = ""
    total_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    evidence: tuple[EvidenceRecord, ...] = ()
    brief: BriefResultRecord = field(default_factory=BriefResultRecord)
    invalid_citations: tuple[str, ...] = ()

    @property
    def ran_calls(self) -> tuple[ToolCallRecord, ...]:
        """Return the calls the loop ran, in the order the model made them."""
        return tuple(call for call in self.tool_calls if call.ran)

    @property
    def brief_text(self) -> str:
        """Return everything the brief said, as one checkable text."""
        parts = [self.brief.summary]
        parts.extend(claim.text for claim in self.brief.claims)
        parts.extend(self.brief.gaps)
        return "\n".join(part for part in parts if part)


@dataclass(frozen=True)
class BriefGrade:
    """One grader's decision about one trajectory."""

    name: str
    outcome: str
    detail: str = ""

    @property
    def passed(self) -> bool:
        """Report whether the grader found nothing wrong."""
        return self.outcome == PASSED

    def to_dict(self) -> dict[str, Any]:
        """Return the grade as a report stores it."""
        return {"name": self.name, "outcome": self.outcome, "detail": self.detail}


def _held_texts(trajectory: BriefTrajectory) -> list[str]:
    """Return every held passage's text, in admission order."""
    return [item.content for item in trajectory.evidence]


def grade_evidence_coverage(task: Any, trajectory: BriefTrajectory) -> BriefGrade:
    """Check that the held evidence reaches every expected passage."""
    expected = list(task.expected_evidence)
    if not expected:
        return BriefGrade(
            "evidence_coverage", UNKNOWN, "the task names no expected evidence"
        )
    held = _held_texts(trajectory)
    missing = [
        item.snippet
        for item in expected
        if not any(contains_snippet(text, item.snippet) for text in held)
    ]
    if not missing:
        return BriefGrade(
            "evidence_coverage", PASSED, f"{len(expected)} expected passages held"
        )
    return BriefGrade(
        "evidence_coverage",
        FAILED,
        "no held passage holds: " + ", ".join(missing),
    )


def grade_both_docs(task: Any, trajectory: BriefTrajectory) -> BriefGrade:
    """Check that a cross-document task cites evidence from both documents."""
    if not task.expects.needs_both_docs:
        return BriefGrade(
            "both_documents", UNKNOWN, "the task needs no second document"
        )
    by_document: dict[str, list[str]] = {}
    for item in trajectory.evidence:
        by_document.setdefault(item.document, []).append(item.content)
    wanted = {item.document for item in task.expected_evidence}
    # A passage from A cannot satisfy B's snippet: each document's snippets
    # are checked only against the passages held from that same document.
    held = set()
    for item in task.expected_evidence:
        texts = by_document.get(item.document, [])
        if any(contains_snippet(text, item.snippet) for text in texts):
            held.add(item.document)
    if wanted <= held:
        return BriefGrade(
            "both_documents", PASSED, f"evidence held from {len(held)} documents"
        )
    return BriefGrade(
        "both_documents",
        FAILED,
        "evidence held from "
        + (", ".join(sorted(held)) or "no document")
        + "; the task needs "
        + ", ".join(sorted(wanted)),
    )


def grade_correctness(task: Any, trajectory: BriefTrajectory) -> BriefGrade:
    """Check that the brief says what the task's reference answer says."""
    if task.expects.expects_abstention:
        return BriefGrade(
            "correctness", UNKNOWN, "the task expects an abstention, not an answer"
        )
    expected = content_words(task.expected_answer)
    if not expected:
        return BriefGrade(
            "correctness", UNKNOWN, "the task declares no expected answer"
        )
    if trajectory.brief.abstained or trajectory.outcome == "abstained":
        return BriefGrade("correctness", FAILED, "the brief abstained")
    found = content_words(trajectory.brief_text)
    overlap = len(expected & found) / len(expected)
    if overlap >= CORRECTNESS_OVERLAP:
        return BriefGrade("correctness", PASSED, f"{overlap:.0%} of the answer reused")
    return BriefGrade("correctness", FAILED, f"only {overlap:.0%} of the answer reused")


def grade_citation_precision(trajectory: BriefTrajectory) -> BriefGrade:
    """Check that every citation names evidence this run collected."""
    cited: list[str] = []
    for claim in trajectory.brief.claims:
        cited.extend(claim.supports)
        cited.extend(claim.conflicts)
    if not cited:
        return BriefGrade(
            "citation_precision", UNKNOWN, "the brief made no citations to check"
        )
    held = {item.evidence_id.strip().upper() for item in trajectory.evidence}
    bad = [name for name in cited if name.strip().upper() not in held]
    if trajectory.invalid_citations:
        bad = sorted(set(bad) | {name for name in trajectory.invalid_citations})
    if not bad:
        return BriefGrade("citation_precision", PASSED, f"{len(cited)} citations held")
    return BriefGrade(
        "citation_precision",
        FAILED,
        "citations naming nothing held: " + ", ".join(bad),
    )


def grade_citation_recall(trajectory: BriefTrajectory) -> BriefGrade:
    """Check that every claim carries a citation a reader can follow."""
    if not trajectory.brief.claims:
        return BriefGrade(
            "citation_recall", UNKNOWN, "the brief made no claims to check"
        )
    held = {item.evidence_id.strip().upper() for item in trajectory.evidence}
    uncited = [
        claim.text[:60]
        for claim in trajectory.brief.claims
        if not any(
            name.strip().upper() in held for name in (*claim.supports, *claim.conflicts)
        )
    ]
    if not uncited:
        return BriefGrade(
            "citation_recall",
            PASSED,
            f"{len(trajectory.brief.claims)} claims cite held evidence",
        )
    return BriefGrade(
        "citation_recall",
        FAILED,
        "claims citing no held evidence: " + "; ".join(uncited),
    )


def grade_page_citation(task: Any, trajectory: BriefTrajectory) -> BriefGrade:
    """
    Check that a page-citation task cites pages a reader can open.

    Every citation must carry a page, and where the task declares the page
    a passage sits on, the held evidence covering it must sit on the same
    page: the right text on the wrong page sends the reader astray.
    """
    if not task.expects.needs_page_citation:
        return BriefGrade("page_citation", UNKNOWN, "the task needs no exact page")
    if not trajectory.brief.claims:
        return BriefGrade(
            "page_citation", FAILED, "the brief made no claims to carry a page"
        )
    pages = {
        item.evidence_id.strip().upper(): item.page for item in trajectory.evidence
    }
    pageless = [
        name
        for claim in trajectory.brief.claims
        for name in (*claim.supports, *claim.conflicts)
        if pages.get(name.strip().upper()) is None
    ]
    if pageless:
        return BriefGrade(
            "page_citation",
            FAILED,
            "cited evidence with no page: " + ", ".join(sorted(set(pageless))),
        )
    wrong: list[str] = []
    for item in task.expected_evidence:
        if getattr(item, "page", None) is None:
            continue
        holders = [
            held.evidence_id
            for held in trajectory.evidence
            if held.document == item.document
            and contains_snippet(held.content, item.snippet)
        ]
        if not holders:
            continue
        if all(pages.get(name.strip().upper()) != item.page for name in holders):
            held_pages = sorted(
                {str(pages.get(name.strip().upper())) for name in holders}
            )
            wrong.append(
                f"{item.document} expects page {item.page} "
                f"but the held evidence sits on {', '.join(held_pages)}"
            )
    if wrong:
        return BriefGrade("page_citation", FAILED, "; ".join(wrong))
    return BriefGrade("page_citation", PASSED, "every citation carries its page")


def grade_conflict_handling(task: Any, trajectory: BriefTrajectory) -> BriefGrade:
    """Check that disagreements stay contested and agreements stay supported."""
    if not trajectory.brief.claims:
        if task.expects.expects_conflict:
            return BriefGrade(
                "conflict_handling", FAILED, "the brief made no claims to contest"
            )
        return BriefGrade(
            "conflict_handling", UNKNOWN, "the brief made no claims to check"
        )
    contested = [
        claim for claim in trajectory.brief.claims if claim.status == "contested"
    ]
    if task.expects.expects_conflict:
        if contested:
            return BriefGrade(
                "conflict_handling",
                PASSED,
                f"{len(contested)} claim(s) kept contested",
            )
        return BriefGrade(
            "conflict_handling",
            FAILED,
            "the task disagrees and no claim is contested",
        )
    if contested:
        return BriefGrade(
            "conflict_handling",
            FAILED,
            f"{len(contested)} claim(s) contested where the task agrees",
        )
    return BriefGrade("conflict_handling", PASSED, "no agreement averaged away")


def grade_gaps(task: Any, trajectory: BriefTrajectory) -> BriefGrade:
    """Check that a question the pair cannot settle says so explicitly."""
    if not task.expects.expects_gap:
        return BriefGrade("gaps", UNKNOWN, "the task expects no gap")
    unresolved = [
        claim for claim in trajectory.brief.claims if claim.status == "unresolved"
    ]
    if trajectory.brief.gaps or unresolved:
        return BriefGrade(
            "gaps",
            PASSED,
            f"{len(trajectory.brief.gaps)} gap(s), {len(unresolved)} unresolved",
        )
    return BriefGrade(
        "gaps", FAILED, "the task is missing evidence and no gap is recorded"
    )


def grade_abstention(task: Any, trajectory: BriefTrajectory) -> BriefGrade:
    """Check that the brief held its tongue exactly where the task asked."""
    abstained = bool(trajectory.brief.abstained) or trajectory.outcome == "abstained"
    if task.expects.expects_abstention:
        if abstained:
            return BriefGrade("abstention", PASSED, "abstained as the task required")
        return BriefGrade(
            "abstention",
            FAILED,
            f"the task is unanswerable and the run ended {trajectory.outcome}",
        )
    if abstained:
        return BriefGrade(
            "abstention",
            FAILED,
            "the brief abstained on a question the pair answers",
        )
    if trajectory.outcome in ("complete", "partial"):
        return BriefGrade("abstention", PASSED, "answered as the task required")
    return BriefGrade(
        "abstention",
        FAILED,
        f"the run ended {trajectory.outcome}, which is neither answer nor abstention",
    )


def grade_tool_validity(trajectory: BriefTrajectory) -> BriefGrade:
    """
    Check that every call named a real tool with fitting arguments.

    Calls the loop already refused are reported by reason, and calls that
    ran are validated against the declared schemas, so a malformed call
    the loop failed to refuse still fails here rather than passing quietly.
    """
    from services.brief.tools import TOOL_SPECS, validate_arguments

    specs = {spec.name: spec for spec in TOOL_SPECS}
    bad: list[str] = []
    for call in trajectory.tool_calls:
        if call.refusal in ("unknown_tool", "invalid_arguments"):
            bad.append(call.tool)
            continue
        if not call.ran:
            continue
        spec = specs.get(call.tool)
        if spec is None or validate_arguments(spec, call.arguments or {}) is not None:
            bad.append(call.tool)
    if bad:
        return BriefGrade(
            "tool_validity",
            FAILED,
            f"{len(bad)} invalid call(s): " + ", ".join(bad),
        )
    return BriefGrade(
        "tool_validity", PASSED, f"{len(trajectory.tool_calls)} calls valid"
    )


def grade_scope(trajectory: BriefTrajectory) -> BriefGrade:
    """Check that the model never reached outside the selected pair."""
    bad = [
        call.tool
        for call in trajectory.tool_calls
        if call.refusal in ("out_of_scope", "unknown_evidence", "unknown_page")
    ]
    if bad:
        return BriefGrade(
            "scope_enforcement",
            FAILED,
            f"{len(bad)} out-of-scope attempt(s): " + ", ".join(bad),
        )
    return BriefGrade("scope_enforcement", PASSED, "no call left the selected pair")


def grade_repeats(trajectory: BriefTrajectory) -> BriefGrade:
    """Check that the model did not repeat a call it had already made."""
    repeats = sum(
        1 for call in trajectory.tool_calls if call.refusal == "repeated_call"
    )
    if repeats:
        return BriefGrade(
            "repeated_calls", FAILED, f"{repeats} repeated call(s) refused"
        )
    return BriefGrade("repeated_calls", PASSED, "no call repeated")


def grade_waste(trajectory: BriefTrajectory) -> BriefGrade:
    """Check that the model stopped searching once it had enough to cite."""
    ran = trajectory.ran_calls
    if len(ran) > MAX_RAN_CALLS:
        return BriefGrade(
            "unnecessary_calls",
            FAILED,
            f"{len(ran)} calls ran against a bound of {MAX_RAN_CALLS}",
        )
    searches: dict[str, int] = {}
    for call in ran:
        if call.tool == "search_passages":
            label = str((call.arguments or {}).get("label", "?"))
            searches[label] = searches.get(label, 0) + 1
    over = {
        label: count
        for label, count in searches.items()
        if count > MAX_SEARCHES_PER_LABEL
    }
    if over:
        return BriefGrade(
            "unnecessary_calls",
            FAILED,
            "searches past the second per document: "
            + ", ".join(f"{label} x{count}" for label, count in sorted(over.items())),
        )
    return BriefGrade("unnecessary_calls", PASSED, f"{len(ran)} calls ran")


def grade_completion(task: Any, trajectory: BriefTrajectory) -> BriefGrade:
    """Check that the brief ended the way its task allows it to end."""
    if trajectory.outcome == "complete":
        return BriefGrade("completion", PASSED, "the brief completed")
    if trajectory.outcome == "abstained" and task.expects.expects_abstention:
        return BriefGrade("completion", PASSED, "abstained as the task required")
    if trajectory.outcome == "partial" and (
        task.expects.expects_gap or task.expects.expects_abstention
    ):
        return BriefGrade("completion", PASSED, "partial where the pair runs out")
    if trajectory.outcome in ("failed", "timeout", "cancelled"):
        return BriefGrade(
            "completion",
            FAILED,
            f"the brief ended {trajectory.outcome}"
            + (
                f" ({trajectory.failure_category})"
                if trajectory.failure_category
                else ""
            ),
        )
    return BriefGrade("completion", FAILED, f"the brief ended {trajectory.outcome}")


OUTCOME_GRADERS = (
    grade_evidence_coverage,
    grade_both_docs,
    grade_correctness,
    grade_citation_precision,
    grade_citation_recall,
    grade_page_citation,
    grade_conflict_handling,
    grade_gaps,
    grade_abstention,
)

TRAJECTORY_GRADERS = (
    grade_tool_validity,
    grade_scope,
    grade_repeats,
    grade_waste,
    grade_completion,
)


def grade_trajectory(task: Any, trajectory: BriefTrajectory) -> dict[str, BriefGrade]:
    """Run every deterministic grader over one brief trial."""
    grades: dict[str, BriefGrade] = {}
    grades["evidence_coverage"] = grade_evidence_coverage(task, trajectory)
    grades["both_documents"] = grade_both_docs(task, trajectory)
    grades["correctness"] = grade_correctness(task, trajectory)
    grades["citation_precision"] = grade_citation_precision(trajectory)
    grades["citation_recall"] = grade_citation_recall(trajectory)
    grades["page_citation"] = grade_page_citation(task, trajectory)
    grades["conflict_handling"] = grade_conflict_handling(task, trajectory)
    grades["gaps"] = grade_gaps(task, trajectory)
    grades["abstention"] = grade_abstention(task, trajectory)
    grades["tool_validity"] = grade_tool_validity(trajectory)
    grades["scope_enforcement"] = grade_scope(trajectory)
    grades["repeated_calls"] = grade_repeats(trajectory)
    grades["unnecessary_calls"] = grade_waste(trajectory)
    grades["completion"] = grade_completion(task, trajectory)
    return grades


__all__ = [
    "BriefClaimRecord",
    "BriefGrade",
    "BriefResultRecord",
    "BriefTrajectory",
    "EvidenceRecord",
    "ToolCallRecord",
    "grade_trajectory",
]
