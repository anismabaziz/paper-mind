"""
Answers a person labelled, so a judge's disagreements are measurable.

A model judge is trusted with the two questions reading cannot settle, and that
trust is only reasonable if it is checked. The set in ``calibration.json`` is a
handful of answers a person labelled by reading each one against its context,
covering all three verdicts and the one case where no verdict is possible: an
answer the context does not settle. A judge that cannot read its own verdict
should land on that case, and a judge that reads a wrong answer as faithful has
said something the run reports.

Agreement is measured only over the cases the judge decided. An undecided case
is counted and left out of the ratio, because a judge that returns Unknown for
everything would otherwise score perfect agreement by refusing to grade.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.answers.judge import (
    RUBRIC_VERSION,
    UNKNOWN,
    VERDICTS,
    judge_faithfulness,
)

CALIBRATION_PATH = Path(__file__).parent.parent / "calibration.json"


@dataclass(frozen=True)
class CalibrationCase:
    """One labelled answer, with the context a judge is asked about it."""

    id: str
    question: str
    context: str
    answer: str
    human_verdict: str


def load_calibration(path: Path = CALIBRATION_PATH) -> tuple[CalibrationCase, ...]:
    """
    Read the labelled calibration set from disk.

    A label the rubric does not define, or a second case with the same id, is
    rejected here rather than quietly graded: a set whose own bookkeeping is
    wrong cannot calibrate anything.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = payload.get("rubric_version")
    if version != RUBRIC_VERSION:
        raise ValueError(
            f"calibration set was written for rubric {version!r}, this run uses "
            f"{RUBRIC_VERSION!r}"
        )
    cases: list[CalibrationCase] = []
    seen: set[str] = set()
    for entry in payload["cases"]:
        if entry["human_verdict"] not in VERDICTS:
            raise ValueError(
                f"{entry['id']} is labelled {entry['human_verdict']!r}, which the "
                f"rubric does not define"
            )
        if entry["id"] in seen:
            raise ValueError(f"{entry['id']} appears twice in the calibration set")
        seen.add(entry["id"])
        cases.append(
            CalibrationCase(
                id=entry["id"],
                question=entry["question"],
                context=entry["context"],
                answer=entry["answer"],
                human_verdict=entry["human_verdict"],
            )
        )
    return tuple(cases)


@dataclass
class CalibrationReport:
    """How one judge read the labelled set, case by case and in total."""

    rubric_version: str
    cases: list[dict[str, Any]] = field(default_factory=list)

    @property
    def agreed(self) -> int:
        """Return the cases where the judge decided, and matched the label."""
        return sum(1 for row in self.cases if row["agreed"])

    @property
    def disagreed(self) -> int:
        """Return the cases where the judge decided, and decided differently."""
        return sum(1 for row in self.cases if not row["agreed"] and not row["unknown"])

    @property
    def unknown(self) -> int:
        """Return the cases the judge could not decide."""
        return sum(1 for row in self.cases if row["unknown"])

    def agreement(self) -> float | None:
        """
        Return the share of decided cases the judge matched.

        None when the judge decided nothing, which is not perfect agreement: it
        is a judge that did not grade.
        """
        decided = self.agreed + self.disagreed
        return self.agreed / decided if decided else None

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a run record stores it."""
        return {
            "rubric_version": self.rubric_version,
            "cases": len(self.cases),
            "agreed": self.agreed,
            "disagreed": self.disagreed,
            "unknown": self.unknown,
            "agreement": self.agreement(),
            "verdicts": self.cases,
        }


def run_calibration(
    cases: Sequence[CalibrationCase], judge_fn: Callable[[str], str]
) -> CalibrationReport:
    """Grade the labelled set with one judge and report where it differed."""
    report = CalibrationReport(rubric_version=RUBRIC_VERSION)
    for case in cases:
        verdict, _ = judge_faithfulness(
            case.question, case.answer, case.context, judge_fn
        )
        undecided = verdict == UNKNOWN
        matched = verdict == case.human_verdict
        report.cases.append(
            {
                "id": case.id,
                "human_verdict": case.human_verdict,
                "judge_verdict": verdict,
                "unknown": undecided,
                # A case the person could not settle is not one the judge can be
                # right about, so it counts as undecided whichever way the judge
                # answered. The row still shows what each of them said.
                "agreed": matched and not undecided,
            }
        )
    return report
