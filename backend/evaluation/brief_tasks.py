"""
The versioned Research Brief task set, and what a run is measured against.

A brief answers across two documents, so a single-document case cannot
describe it. Each task names the pair the brief may read, the question it is
asked, the passages that answer it, and what the brief has to do with them:
whether both documents are needed, whether the evidence disagrees, whether
part of the question has no answer in the pair, whether the brief has to hold
its tongue, and whether the citation has to name an exact page.

Like the labeled case set, the set is versioned by directory: a new set is a
new directory and the one it replaces stays readable beside it. Every task id
is a stable slug, every document is pinned by content hash, and every problem
with the file is reported together rather than one at a time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TASKS_DIR = Path(__file__).parent / "datasets"

CURRENT_BRIEF_TASKS = "2026-09-brief-tasks-v1"

TASKS_PATH = TASKS_DIR / CURRENT_BRIEF_TASKS / "tasks.json"

CROSS_LOOKUP = "cross_lookup"
AGREEMENT = "agreement"
DISAGREEMENT = "disagreement"
MISSING_EVIDENCE = "missing_evidence"
UNANSWERABLE = "unanswerable"
PAGE_CITATION = "page_citation"
CATEGORIES = (
    CROSS_LOOKUP,
    AGREEMENT,
    DISAGREEMENT,
    MISSING_EVIDENCE,
    UNANSWERABLE,
    PAGE_CITATION,
)


class BriefTasksInvalid(ValueError):
    """Everything wrong with a brief task set, raised together."""

    def __init__(self, problems: list[str]):
        """Report every problem found, in the order they were found."""
        self.problems = list(problems)
        super().__init__(
            f"the brief task set has {len(self.problems)} problems:\n"
            + "\n".join(f"  - {problem}" for problem in self.problems)
        )


@dataclass(frozen=True)
class ExpectedEvidence:
    """One passage a brief has to be able to cite, and the document holding it."""

    document: str
    snippet: str
    page: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the expectation as the task file stores it."""
        stored: dict[str, Any] = {"document": self.document, "snippet": self.snippet}
        if self.page is not None:
            stored["page"] = self.page
        return stored


@dataclass(frozen=True)
class TaskExpects:
    """What the brief has to do with the evidence, beyond quoting it."""

    needs_both_docs: bool = False
    expects_conflict: bool = False
    expects_gap: bool = False
    expects_abstention: bool = False
    needs_page_citation: bool = False
    needs_compare: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the expectations as the task file stores them."""
        return {
            "needs_both_docs": self.needs_both_docs,
            "expects_conflict": self.expects_conflict,
            "expects_gap": self.expects_gap,
            "expects_abstention": self.expects_abstention,
            "needs_page_citation": self.needs_page_citation,
            "needs_compare": self.needs_compare,
        }


@dataclass(frozen=True)
class BriefTask:
    """One reviewed cross-document question with its grading contract."""

    id: str
    documents: tuple[str, str]
    category: str
    question: str
    expected_answer: str
    expected_evidence: tuple[ExpectedEvidence, ...]
    rubric: str
    expects: TaskExpects
    absent_terms: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the task as the task file stores it."""
        stored: dict[str, Any] = {
            "id": self.id,
            "documents": list(self.documents),
            "category": self.category,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "expected_evidence": [item.to_dict() for item in self.expected_evidence],
            "rubric": self.rubric,
            "expects": self.expects.to_dict(),
        }
        if self.absent_terms:
            stored["absent_terms"] = list(self.absent_terms)
        return stored


@dataclass(frozen=True)
class BriefTaskSet:
    """One version of the brief task set with the documents its tasks need."""

    version: str
    reviewed_on: str
    documents: tuple[dict[str, Any], ...]
    tasks: tuple[BriefTask, ...]

    def task(self, task_id: str) -> BriefTask:
        """Return one task by its id."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise KeyError(f"{task_id} is not a task in this set")

    def counts_by_category(self) -> dict[str, int]:
        """Return how many tasks each kind of question is asked."""
        counts: dict[str, int] = {}
        for task in self.tasks:
            counts[task.category] = counts.get(task.category, 0) + 1
        return counts


def load_brief_tasks(path: Path = TASKS_PATH) -> BriefTaskSet:
    """Read one version of the brief task set, or explain why it cannot be read."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_payload(payload)


def validate_payload(payload: dict[str, Any]) -> BriefTaskSet:
    """Turn a loaded brief task set into tasks, or refuse it with every reason."""
    problems: list[str] = []
    documents = _documents(payload, problems)
    known = {str(entry.get("filename", "")) for entry in documents}
    tasks = _tasks(payload, known, problems)
    version = _text(payload.get("version"))
    reviewed_on = _text(payload.get("reviewed_on"))
    if not version:
        problems.append("the set declares no version")
    if not reviewed_on:
        problems.append("the set records no date it was reviewed on")
    _check_coverage(tasks, problems)
    if problems:
        raise BriefTasksInvalid(problems)
    return BriefTaskSet(
        version=version,
        reviewed_on=reviewed_on,
        documents=tuple(documents),
        tasks=tuple(tasks),
    )


def _documents(payload: dict[str, Any], problems: list[str]) -> list[dict[str, Any]]:
    """Read the document entries, checking each one pins licensed bytes."""
    documents: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in payload.get("documents") or []:
        if not isinstance(entry, dict):
            problems.append("a document entry is not an object")
            continue
        filename = _text(entry.get("filename"))
        if not filename:
            problems.append("a document entry declares no filename")
            continue
        if filename in seen:
            problems.append(f"{filename} is declared twice in the document list")
            continue
        seen.add(filename)
        for name in ("version", "sha256", "origin", "license"):
            if not _text(entry.get(name)):
                problems.append(f"{filename} declares no {name}")
        documents.append(entry)
    if not documents:
        problems.append("the set declares no documents")
    return documents


def _tasks(
    payload: dict[str, Any], known: set[str], problems: list[str]
) -> list[BriefTask]:
    """Read the tasks, checking each one declares what its category requires."""
    tasks: list[BriefTask] = []
    seen: set[str] = set()
    for entry in payload.get("tasks") or []:
        task = _task(entry, problems)
        if task is None:
            continue
        if task.id in seen:
            problems.append(f"{task.id} appears twice in the set")
            continue
        seen.add(task.id)
        for filename in task.documents:
            if filename not in known:
                problems.append(
                    f"{task.id} reads {filename}, which the set does not ship"
                )
        tasks.append(task)
    return tasks


def _task(entry: Any, problems: list[str]) -> BriefTask | None:
    """Read one task, or report everything missing from it and return nothing."""
    if not isinstance(entry, dict):
        problems.append("a task entry is not an object")
        return None
    task_id = _text(entry.get("id"))
    documents = entry.get("documents")
    pair: tuple[str, str] = ("", "")
    if (
        isinstance(documents, list)
        and len(documents) == 2
        and all(isinstance(name, str) and name.strip() for name in documents)
    ):
        pair = (documents[0].strip(), documents[1].strip())
    else:
        problems.append(
            f"{task_id or 'a task'} names no pair of documents; a brief reads exactly two"
        )
    category = _text(entry.get("category"))
    if category not in CATEGORIES:
        problems.append(
            f"{task_id} is category {category!r}; the categories are "
            f"{', '.join(CATEGORIES)}"
        )
    if not _text(entry.get("question")):
        problems.append(f"{task_id} asks nothing")
    if not _text(entry.get("rubric")):
        problems.append(f"{task_id} declares no rubric a person could check")
    expected = entry.get("expected_evidence") or []
    evidence: list[ExpectedEvidence] = []
    if isinstance(expected, list):
        for item in expected:
            if not isinstance(item, dict):
                problems.append(f"{task_id} names evidence that is not an object")
                continue
            document = _text(item.get("document"))
            snippet = _text(item.get("snippet"))
            if not document or not snippet:
                problems.append(
                    f"{task_id} names evidence with no document or no snippet"
                )
                continue
            page = item.get("page")
            if page is not None and (
                not isinstance(page, int) or isinstance(page, bool) or page < 1
            ):
                problems.append(f"{task_id} names evidence with no valid page")
                continue
            evidence.append(
                ExpectedEvidence(document=document, snippet=snippet, page=page)
            )
    expects_entry = entry.get("expects") or {}
    expects = TaskExpects(
        needs_both_docs=bool(expects_entry.get("needs_both_docs", False)),
        expects_conflict=bool(expects_entry.get("expects_conflict", False)),
        expects_gap=bool(expects_entry.get("expects_gap", False)),
        expects_abstention=bool(expects_entry.get("expects_abstention", False)),
        needs_page_citation=bool(expects_entry.get("needs_page_citation", False)),
        needs_compare=bool(expects_entry.get("needs_compare", False)),
    )
    _check_task_expectations(task_id, category, evidence, expects, entry, problems)
    absent = tuple(
        item.strip()
        for item in (entry.get("absent_terms") or [])
        if isinstance(item, str) and item.strip()
    )
    return BriefTask(
        id=task_id,
        documents=pair,
        category=category,
        question=_text(entry.get("question")),
        expected_answer=_text(entry.get("expected_answer")),
        expected_evidence=tuple(evidence),
        rubric=_text(entry.get("rubric")),
        expects=expects,
        absent_terms=absent,
    )


def _check_task_expectations(
    task_id: str,
    category: str,
    evidence: list[ExpectedEvidence],
    expects: TaskExpects,
    entry: dict[str, Any],
    problems: list[str],
) -> None:
    """Check a task declares what its own category has to declare."""
    if category == UNANSWERABLE:
        if evidence:
            problems.append(
                f"{task_id} is unanswerable, so it names no evidence; "
                f"it names {len(evidence)}"
            )
        if not entry.get("absent_terms"):
            problems.append(
                f"{task_id} is unanswerable but names no absent term, so nothing "
                "checks that its documents cannot answer it"
            )
        if not expects.expects_abstention:
            problems.append(
                f"{task_id} is unanswerable but does not expect an abstention"
            )
    elif category == MISSING_EVIDENCE:
        if not expects.expects_gap:
            problems.append(
                f"{task_id} is missing evidence but expects no gap to be recorded"
            )
    elif category == DISAGREEMENT:
        if not expects.expects_conflict:
            problems.append(
                f"{task_id} is a disagreement but expects no conflict to be kept"
            )
    if category != UNANSWERABLE and not evidence:
        problems.append(
            f"{task_id} expects an answer but names no evidence, so coverage "
            "cannot be graded"
        )
    if category != UNANSWERABLE and not _text(entry.get("expected_answer")):
        problems.append(f"{task_id} expects an answer but declares no expected answer")


def _check_coverage(tasks: list[BriefTask], problems: list[str]) -> None:
    """Check the set covers every kind of brief question it claims to balance."""
    counts = {task.category: 0 for task in tasks}
    for task in tasks:
        counts[task.category] = counts.get(task.category, 0) + 1
    for category in CATEGORIES:
        if not counts.get(category):
            problems.append(f"the set holds no {category} task")


def _text(value: Any) -> str:
    """Return a required string field, empty when it is missing or not a string."""
    return value.strip() if isinstance(value, str) else ""


__all__ = [
    "AGREEMENT",
    "CATEGORIES",
    "CROSS_LOOKUP",
    "CURRENT_BRIEF_TASKS",
    "DISAGREEMENT",
    "MISSING_EVIDENCE",
    "PAGE_CITATION",
    "TASKS_PATH",
    "UNANSWERABLE",
    "BriefTask",
    "BriefTaskSet",
    "BriefTasksInvalid",
    "ExpectedEvidence",
    "TaskExpects",
    "load_brief_tasks",
    "validate_payload",
]
