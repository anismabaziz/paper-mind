"""
The labeled case set, and what a run is measured against.

The set is a file a person wrote and checked, so most of this module is about
refusing a set that has not been checked. A case naming a document nobody
shipped, an expected answer with nothing behind it, a question called
unanswerable whose answer is sitting in the document, two cases sharing an id.
Each of those is reported by name rather than loaded, and all of them are
reported at once so one pass tells a reviewer everything that has to change.

Two properties are load-bearing, and both are checked rather than hoped for.
Every document is pinned by content hash, so a regenerated or replaced file
cannot quietly invalidate expectations a person wrote against the old one. And
the cases are split into a tuning half and a reported half before any
configuration is chosen, so the number that gets quoted was not the number
that got fitted.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.graders import content_words
from evaluation.metrics import contains_snippet
from evaluation.outcomes import (
    ABSTAINED,
    ANSWERED,
    CITATION_ERROR,
    PROVIDER_ERROR,
    REQUIRED_OUTCOMES,
)
from services.parsing.pdf_service import PDFParser

DATASETS_DIR = Path(__file__).parent / "datasets"

#: Where the documents the cases are labelled against are committed.
SAMPLE_DOCS_DIR = Path(__file__).parent / "sample_docs"

#: The version a run measures by default. It is a directory name, so a new set
#: is a new directory and the one it replaces stays readable beside it.
CURRENT_DATASET = "2026-09-labeled-cases-v1"

DATASET_PATH = DATASETS_DIR / CURRENT_DATASET / "dataset.json"

#: The two splits, named for what a run does with them. The tuning split is
#: read while a configuration is being chosen; the reported split is read once
#: and quoted.
TUNING = "tuning"
REPORTED = "validation"
SPLITS = (TUNING, REPORTED)

#: How a case is asked, which is what the set is balanced over.
EXACT_LOOKUP = "exact_lookup"
PARAPHRASE = "paraphrase"
NUMERIC = "numeric_reasoning"
TABLE = "table_reasoning"
MULTI_SECTION = "multi_section_evidence"
UNANSWERABLE = "unanswerable"
FOLLOW_UP = "follow_up"
PROMPT_INJECTION = "prompt_injection"
CITATION_VALIDATION = "citation_validation"
PROVIDER_FAILURE = "provider_failure"
CATEGORIES = (
    EXACT_LOOKUP,
    PARAPHRASE,
    NUMERIC,
    TABLE,
    MULTI_SECTION,
    UNANSWERABLE,
    FOLLOW_UP,
    PROMPT_INJECTION,
    CITATION_VALIDATION,
    PROVIDER_FAILURE,
)

#: A failure a run can be made to produce, and the outcome the case declares
#: must come out of it. A case with a fault has no nominal run: the fault is
#: the case, and what it checks is that the failure is reported as itself
#: rather than as an answer.
FAULTS = {
    "provider_unavailable": PROVIDER_ERROR,
    "invalid_citation": CITATION_ERROR,
}

#: A set outside this range was either not reviewed to the standard the rest of
#: the set was, or was trimmed for one run and needs a version of its own.
MIN_CASES = 40
MAX_CASES = 60

#: How much of a piece of expected evidence a case is allowed to restate in its
#: own question. A question built out of the passage it is meant to be answered
#: from can be passed by a lexical match, which measures the matcher rather than
#: the reader.
MAX_QUOTED_EVIDENCE = 0.8

#: The categories whose cases claim to need the passage read rather than
#: matched. Exact lookup is the one category that reuses the document's wording
#: on purpose, so it is the one exempt.
NEEDS_READING = tuple(category for category in CATEGORIES if category != EXACT_LOOKUP)

#: A case id appears in reports that are compared across runs, so it is a
#: stable slug rather than anything that can be renumbered.
_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


class DatasetInvalid(ValueError):
    """
    Everything wrong with a case set, raised together.

    A reviewer fixing a set wants the whole list rather than the first thing
    the loader noticed, so the problems are collected and reported in one go.
    """

    def __init__(self, problems: Sequence[str]):
        """Report every problem found, in the order they were found."""
        self.problems = list(problems)
        super().__init__(
            f"the labeled case set has {len(self.problems)} problems:\n"
            + "\n".join(f"  - {problem}" for problem in self.problems)
        )


def content_hash(path: Path) -> str:
    """Return the hash that pins one document's bytes to the labels on them."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class SourceDocument:
    """
    One document cases are labelled against, pinned to the bytes reviewed.

    The hash is the point. Without it a regenerated file is a new document
    under the same name, and every expectation a person wrote against the old
    one keeps passing validation while measuring something else.
    """

    filename: str
    version: str
    sha256: str
    origin: str
    license: str

    def to_dict(self) -> dict[str, Any]:
        """Return the document as the dataset file stores it."""
        return {
            "filename": self.filename,
            "version": self.version,
            "sha256": self.sha256,
            "origin": self.origin,
            "license": self.license,
        }


@dataclass(frozen=True)
class Case:
    """
    One reviewed question, with everything a grader may read about it.

    A case carries its own expectations rather than letting the run infer them.
    It says what the answer is worth, which passages have to support it, which
    outcome the run has to end as, and what a person checking the answer should
    find. A grader handed an expectation it was told separately could disagree
    with the set about what the case asked.
    """

    id: str
    document: str
    split: str
    category: str
    question: str
    expected_outcome: str
    expected_answer: str | None = None
    expected_evidence: tuple[str, ...] = ()
    rubric: str = ""
    absent_terms: tuple[str, ...] = ()
    follow_up: tuple[str, ...] = ()
    fault: str | None = None
    notes: str = ""

    @property
    def faulty(self) -> bool:
        """Report whether this case only makes sense in a run with a fault."""
        return self.fault is not None

    def to_dict(self) -> dict[str, Any]:
        """Return the case as the dataset file stores it."""
        stored: dict[str, Any] = {
            "id": self.id,
            "document": self.document,
            "split": self.split,
            "category": self.category,
            "question": self.question,
            "expected_outcome": self.expected_outcome,
        }
        for name, value in (
            ("expected_answer", self.expected_answer),
            ("expected_evidence", self.expected_evidence or None),
            ("rubric", self.rubric or None),
            ("absent_terms", self.absent_terms or None),
            ("follow_up", self.follow_up or None),
            ("fault", self.fault),
            ("notes", self.notes or None),
        ):
            if value:
                stored[name] = list(value) if isinstance(value, tuple) else value
        return stored


@dataclass(frozen=True)
class Dataset:
    """
    One version of the case set, with the documents its cases are about.

    Selecting cases is a read. Asking for the reported split returns those
    cases and holds back the ones that only mean anything when a fault is
    injected, because a reported number that counted a deliberately broken
    provider would be reporting the test rather than the system.
    """

    version: str
    reviewed_on: str
    documents: tuple[SourceDocument, ...]
    cases: tuple[Case, ...]

    def document(self, filename: str) -> SourceDocument:
        """Return the document cases are labelled against."""
        for document in self.documents:
            if document.filename == filename:
                return document
        raise KeyError(f"{filename} is not a document in this set")

    def cases_for(
        self, split: str, *, include_faults: bool = False
    ) -> tuple[Case, ...]:
        """
        Return the cases of one split, in the order the file lists them.

        A case carrying a fault is left out unless it is asked for: it has no
        answer to grade, so a run that counted it would be reporting an
        injected failure as a quality result.
        """
        if split not in SPLITS:
            raise ValueError(f"{split!r} is not a split; use {', '.join(SPLITS)}")
        return tuple(
            case
            for case in self.cases
            if case.split == split and (include_faults or not case.faulty)
        )

    def faults_for(self, split: str) -> tuple[Case, ...]:
        """Return the cases of one split that need a fault injected to run."""
        return tuple(
            case for case in self.cases_for(split, include_faults=True) if case.faulty
        )

    def counts_by_category(self) -> dict[str, int]:
        """Return how many cases each way of asking a question is asked."""
        return _counts(self.cases, lambda case: case.category)

    def counts_by_split(self) -> dict[str, int]:
        """Return how many cases each split holds."""
        return _counts(self.cases, lambda case: case.split)

    def count_in(self, split: str) -> int:
        """Return how many cases of a split a run would be asked."""
        return len(self.cases_for(split))

    def to_dict(self) -> dict[str, Any]:
        """Return the dataset as the file stores it."""
        return {
            "version": self.version,
            "reviewed_on": self.reviewed_on,
            "documents": [document.to_dict() for document in self.documents],
            "cases": [case.to_dict() for case in self.cases],
        }


def load_dataset(path: Path = DATASET_PATH, *, docs_dir: Path | None = None) -> Dataset:
    """
    Read one version of the case set, or explain why it cannot be read.

    Everything checkable without opening a document is checked here: the
    references, the labels, the split membership, the licensing, the fields
    each category requires, and the content hash of every source document. What
    can only be checked against the documents themselves, such as whether an
    unanswerable case really is unanswerable, is
    :func:`evidence_problems`.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_payload(payload, docs_dir=docs_dir)


def validate_payload(
    payload: dict[str, Any], *, docs_dir: Path | None = None
) -> Dataset:
    """
    Turn a loaded case set into a dataset, or refuse it.

    Every problem is collected before anything is raised, so one pass tells a
    reviewer everything that has to change.
    """
    problems: list[str] = []
    documents = _documents(payload, docs_dir, problems)
    cases = _cases(payload, documents, problems)
    dataset = Dataset(
        version=_text(payload.get("version")),
        reviewed_on=_text(payload.get("reviewed_on")),
        documents=documents,
        cases=cases,
    )
    if not dataset.version:
        problems.append("the set declares no version")
    if not dataset.reviewed_on:
        problems.append("the set records no date it was reviewed on")
    if not MIN_CASES <= len(cases) <= MAX_CASES:
        problems.append(
            f"the set holds {len(cases)} cases; a reviewed set is between "
            f"{MIN_CASES} and {MAX_CASES}"
        )
    _check_coverage(dataset, problems)
    _check_splits(dataset, problems)
    if problems:
        raise DatasetInvalid(problems)
    return dataset


def _documents(
    payload: dict[str, Any], docs_dir: Path | None, problems: list[str]
) -> tuple[SourceDocument, ...]:
    """Read the document entries, checking each one pins real, licensed bytes."""
    documents: list[SourceDocument] = []
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
        missing = [
            name
            for name in ("version", "origin", "license")
            if not _text(entry.get(name))
        ]
        if missing:
            problems.append(f"{filename} declares no {', '.join(missing)}")
        problems.extend(_hash_problems(filename, entry.get("sha256"), docs_dir))
        documents.append(
            SourceDocument(
                filename=filename,
                version=_text(entry.get("version")),
                sha256=_text(entry.get("sha256")),
                origin=_text(entry.get("origin")),
                license=_text(entry.get("license")),
            )
        )
    return tuple(documents)


def _hash_problems(filename: str, declared: Any, docs_dir: Path | None) -> list[str]:
    """Return what is wrong with a document's content pin, if anything is."""
    pinned = _text(declared)
    if not pinned:
        return [f"{filename} pins no content hash"]
    if docs_dir is None:
        return []
    path = Path(docs_dir) / filename
    if not path.is_file():
        return [f"{filename} is pinned but not committed"]
    if content_hash(path) != pinned:
        return [
            f"{filename} no longer matches the content hash its cases were "
            "reviewed against"
        ]
    return []


def _cases(
    payload: dict[str, Any], documents: Sequence[SourceDocument], problems: list[str]
) -> tuple[Case, ...]:
    """Read the cases, checking each one declares what its category requires."""
    known = {document.filename for document in documents}
    cases: list[Case] = []
    seen: set[str] = set()
    for entry in payload.get("cases") or []:
        case = _case(entry, problems)
        if case is None:
            continue
        if case.id in seen:
            problems.append(f"{case.id} appears twice in the set")
            continue
        seen.add(case.id)
        if case.document not in known:
            problems.append(
                f"{case.id} is labelled against {case.document}, which the set does "
                "not ship"
            )
        cases.append(case)
    return tuple(cases)


def _case(entry: Any, problems: list[str]) -> Case | None:
    """Read one case, or report everything missing from it and return nothing."""
    if not isinstance(entry, dict):
        problems.append(f"a case entry is a {type(entry).__name__}, not an object")
        return None
    case_id = _text(entry.get("id"))
    if not _ID.fullmatch(case_id):
        problems.append(
            f"case id {case_id!r} is not a slug of lowercase words joined by dashes"
        )
    case = Case(
        id=case_id,
        document=_text(entry.get("document")),
        split=_text(entry.get("split")),
        category=_text(entry.get("category")),
        question=_text(entry.get("question")),
        expected_outcome=_text(entry.get("expected_outcome")),
        expected_answer=_optional_text(entry.get("expected_answer")),
        expected_evidence=_strings(entry.get("expected_evidence")),
        rubric=_text(entry.get("rubric")),
        absent_terms=_strings(entry.get("absent_terms")),
        follow_up=_strings(entry.get("follow_up")),
        fault=_optional_text(entry.get("fault")),
        notes=_text(entry.get("notes")),
    )
    if not case.question:
        problems.append(f"{case_id} asks nothing")
    if case.split not in SPLITS:
        problems.append(
            f"{case_id} is in split {case.split!r}; the splits are {', '.join(SPLITS)}"
        )
    if case.category not in CATEGORIES:
        problems.append(
            f"{case_id} is category {case.category!r}; the categories are "
            f"{', '.join(CATEGORIES)}"
        )
    if case.expected_outcome not in REQUIRED_OUTCOMES:
        problems.append(
            f"{case_id} expects outcome {case.expected_outcome!r}, which is not one "
            f"of {', '.join(REQUIRED_OUTCOMES)}"
        )
    if case.fault is not None and case.fault not in FAULTS:
        problems.append(
            f"{case_id} names fault {case.fault!r}; the known faults are "
            f"{', '.join(FAULTS)}"
        )
    _check_expectations(case, problems)
    return case


def _check_expectations(case: Case, problems: list[str]) -> None:
    """Check a case declares everything its own category has to declare."""
    if case.fault is not None:
        problems.extend(_fault_problems(case))
        return
    if case.expected_outcome == ANSWERED and not case.expected_answer:
        problems.append(f"{case.id} expects an answer but declares no expected answer")
    if not case.rubric:
        problems.append(
            f"{case.id} expects an answer a person could check, but declares no rubric"
        )
    if case.category == UNANSWERABLE:
        problems.extend(_unanswerable_problems(case))
    elif case.expected_outcome == ANSWERED and not case.expected_evidence:
        problems.append(
            f"{case.id} expects an answer but names no evidence, so a citation to the "
            "wrong passage cannot be caught"
        )
    if case.category == FOLLOW_UP and not case.follow_up:
        problems.append(f"{case.id} is a follow-up but records no earlier question")
    if case.category in NEEDS_READING:
        for snippet in case.expected_evidence:
            restated = restated_evidence(case.question, snippet)
            if restated > MAX_QUOTED_EVIDENCE:
                problems.append(
                    f"{case.id} restates {restated:.0%} of the evidence it expects in "
                    "its own question, so a lexical match alone could pass it"
                )


def _unanswerable_problems(case: Case) -> list[str]:
    """
    Return what is wrong with a case the document cannot answer.

    A question nothing in the document answers has no evidence to point at, and
    the two honest ways out are both acceptable: the run can abstain before it
    spends a call, or it can answer that it does not know. So the case names
    the terms that would answer it, and says which of the two it expects.
    """
    problems: list[str] = []
    if case.expected_evidence:
        problems.append(
            f"{case.id} is unanswerable, so it names no evidence; it names "
            f"{len(case.expected_evidence)}"
        )
    if not case.absent_terms:
        problems.append(
            f"{case.id} is unanswerable but names no absent term, so nothing checks "
            "that its document cannot answer it"
        )
    if case.expected_outcome not in (ANSWERED, ABSTAINED):
        problems.append(
            f"{case.id} is unanswerable, so the run may abstain or answer that it does "
            f"not know; {case.expected_outcome!r} is neither"
        )
    return problems


def _fault_problems(case: Case) -> list[str]:
    """Return what is wrong with a case that only runs with a fault injected."""
    problems: list[str] = []
    required = FAULTS.get(case.fault or "")
    if required and case.expected_outcome != required:
        problems.append(
            f"{case.id} injects {case.fault!r}, which has to come out as "
            f"{required!r}, not {case.expected_outcome!r}"
        )
    if case.expected_answer or case.expected_evidence:
        problems.append(
            f"{case.id} injects a fault, so it has no answer to grade; it declares "
            "one anyway"
        )
    if case.rubric:
        problems.append(
            f"{case.id} injects a fault, so there is no answer for a rubric to check; "
            "it declares one anyway"
        )
    return problems


def _counts(cases: Sequence[Case], value: Callable[[Case], str]) -> dict[str, int]:
    """Return how many cases carry each of the values one field takes."""
    counts: dict[str, int] = {}
    for case in cases:
        counts[value(case)] = counts.get(value(case), 0) + 1
    return counts


def _check_coverage(dataset: Dataset, problems: list[str]) -> None:
    """Check the set covers every way of failing it claims to be balanced over."""
    counts = dataset.counts_by_category()
    for category in CATEGORIES:
        if not counts.get(category):
            problems.append(f"the set holds no {category} case")


def _check_splits(dataset: Dataset, problems: list[str]) -> None:
    """
    Check both splits are worth reading.

    A reported split smaller than the tuning split is a number decided by a
    handful of questions, and a set where every case is in the tuning split has
    nothing to report at all.
    """
    for split in SPLITS:
        if not dataset.count_in(split):
            problems.append(f"the {split} split is empty")
    tuning = len(dataset.cases_for(TUNING))
    reported = len(dataset.cases_for(REPORTED))
    if reported < tuning:
        problems.append(
            f"the reported split holds {reported} cases against {tuning} tuning "
            "cases, so the reported number is decided by the smaller set"
        )


def abstention_required(case: Case) -> bool | None:
    """
    Return whether the run has to abstain on a case, or may do either.

    Only the two outcomes a question can honestly end as decide this, and a
    question the document cannot answer decides nothing: the run may abstain
    before it spends a call, or answer that it does not know, and both are
    honest. A provider failure or an unusable citation is neither an answer nor
    an abstention either, so the abstention grader reports nothing rather than
    guessing.
    """
    if case.category == UNANSWERABLE:
        return None
    if case.expected_outcome == ABSTAINED:
        return True
    if case.expected_outcome == ANSWERED:
        return False
    return None


def restated_evidence(question: str, snippet: str) -> float:
    """
    Return how much of a piece of expected evidence a question repeats.

    The snippet's content words are counted against the question's. A case
    whose question is built out of the passage it is meant to be answered from
    can be passed by lexical matching alone, which says nothing about whether
    the answer is right.
    """
    expected = content_words(snippet)
    if not expected:
        return 0.0
    return len(expected & content_words(question)) / len(expected)


def evidence_problems(dataset: Dataset, docs_dir: Path) -> list[str]:
    """
    Return everything about the labels that can only be checked against the text.

    Whether an expected passage is really in its document, whether a passage a
    case calls unanswerable really is absent, and whether a case claiming
    evidence from several sections really draws on more than one page are
    questions about the documents rather than about the file, so they cost a
    parse each and are asked once instead of on every load.
    """
    pages = {
        document.filename: PDFParser().extract_pages(
            (Path(docs_dir) / document.filename).read_bytes()
        )
        for document in dataset.documents
    }
    problems: list[str] = []
    for case in dataset.cases:
        pages_of_case = pages.get(case.document, [])
        problems.extend(_evidence_problems(case, pages_of_case))
        for term in case.absent_terms:
            if contains_snippet(" ".join(pages_of_case), term):
                problems.append(
                    f"{case.id} is unanswerable, but {term!r} is in {case.document}, "
                    "so the document can answer it after all"
                )
    return problems


def _evidence_problems(case: Case, pages: Sequence[str]) -> list[str]:
    """Return what is wrong with one case's expected evidence."""
    problems: list[str] = []
    holding: set[int] = set()
    for snippet in case.expected_evidence:
        found = [
            number
            for number, text in enumerate(pages, start=1)
            if contains_snippet(text, snippet)
        ]
        if not found:
            problems.append(
                f"{case.id} expects evidence that is not in {case.document}: {snippet!r}"
            )
        elif len(found) > 1:
            problems.append(
                f"{case.id} expects evidence that appears on pages "
                f"{', '.join(str(number) for number in found)} of {case.document}, so "
                "a page citation could be either"
            )
        holding.update(found)
    if case.category == MULTI_SECTION and len(holding) < 2:
        problems.append(
            f"{case.id} is labelled as needing evidence from several sections, but "
            f"its evidence sits on {len(holding)} page of {case.document}"
        )
    return problems


def _text(value: Any) -> str:
    """Return a required string field, empty when it is missing or not a string."""
    return value.strip() if isinstance(value, str) else ""


def _optional_text(value: Any) -> str | None:
    """Return an optional string field, or None when it was left out."""
    text = _text(value)
    return text or None


def _strings(value: Any) -> tuple[str, ...]:
    """Return a list-of-strings field as a tuple, ignoring anything else."""
    if not isinstance(value, list):
        return ()
    return tuple(
        item.strip() for item in value if isinstance(item, str) and item.strip()
    )
