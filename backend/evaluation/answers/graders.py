"""
The graders that decide an answer without asking a model.

A number a model produced is only worth as much as the model that produced it,
so everything in this module is decided by reading what the run already
recorded: the stored answer, the claims it declared, the Passages the model was
shown, and the outcome the case ended as. The same run therefore always gets the
same grades, and a disagreement can be traced to the exact claim that caused it.

Three outcomes, and the third one matters. A grade is ``passed`` or ``failed``,
or it is ``unknown`` — the grader had nothing to decide. An answer whose claims
cite nothing, in a Document whose Passages carry no page, cannot be graded on
either, and counting it as a pass would inflate the metric with answers nobody
read. Unknown is reported beside the mean rather than folded into it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from evaluation.answers.metrics import contains_snippet
from services.text import STOPWORDS, word_parts

PASSED = "passed"
FAILED = "failed"
UNKNOWN = "unknown"

#: How much of a claim's own wording a cited Passage has to repeat before the
#: citation counts as supporting it. Half the content words is the point where
#: a citation that merely shares a topic stops looking like one that supports
#: the sentence it is attached to.
SUPPORTING_OVERLAP = 0.5

#: A page reference as a reader writes it: "page 4", "pages 4-5", "p. 4".
_PAGE_REFERENCE = re.compile(
    r"\b(?:pages?|pp?\.)\s*(\d+)(?:\s*[-–—]\s*(\d+))?", re.IGNORECASE
)


def content_words(text: str) -> set[str]:
    """Return the words of a text that carry its meaning, folded to lower case."""
    return {
        word
        for word in word_parts(str(text or ""))
        if word not in STOPWORDS and len(word) > 1
    }


def pages_in(text: str) -> set[int]:
    """
    Return the page numbers a text points the reader at.

    Only a number that follows the word "page" counts. A year, a count, and a
    figure are all digits, and an answer about a 2018 study is not claiming to
    be on page 2018.
    """
    found: set[int] = set()
    for first, last in _PAGE_REFERENCE.findall(str(text or "")):
        start = int(first)
        end = int(last) if last else start
        found.update(range(start, max(start, end) + 1))
    return found


@dataclass(frozen=True)
class GradedCase:
    """
    Everything a deterministic grader is allowed to look at.

    The case carries its own expectation rather than being handed one: a grader
    that had to be told separately what to look for could disagree with the run
    about what the case asked. ``requires_abstention`` is None for a case that
    declares nothing about abstaining, which is different from False.
    """

    id: str
    outcome: str
    answer: str | None
    claims: tuple[dict[str, Any], ...]
    sources: tuple[dict[str, Any], ...]
    expected_answer: str | None = None
    expected_evidence: tuple[str, ...] = ()
    requires_abstention: bool | None = False
    expected_outcome: str | None = None

    @property
    def source_texts(self) -> dict[str, str]:
        """Return each supplied Passage's text, keyed by the id the model saw."""
        return {
            str(source.get("source_id")): str(source.get("content") or "")
            for source in self.sources
        }

    @property
    def cited(self) -> tuple[str, ...]:
        """Return the Passage ids the answer's claims named, in order, no repeats."""
        seen: list[str] = []
        for claim in self.claims:
            for source_id in _declared_ids(claim):
                if source_id not in seen:
                    seen.append(source_id)
        return tuple(seen)


@dataclass(frozen=True)
class Grade:
    """One grader's decision about one case, and what it decided on."""

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


def _declared_ids(claim: Any) -> tuple[str, ...]:
    """Return the ids one claim declares, as the app spells them, in order."""
    if not isinstance(claim, dict):
        return ()
    declared = claim.get("sources")
    if not isinstance(declared, list):
        return ()
    ids: list[str] = []
    for source_id in declared:
        if isinstance(source_id, str):
            spelled = source_id.strip().upper()
            if spelled and spelled not in ids:
                ids.append(spelled)
    return tuple(ids)


def _claim_text(claim: Any) -> str:
    """Return one claim's text, or the empty string when it has none."""
    if not isinstance(claim, dict):
        return ""
    text = claim.get("claim")
    return text.strip() if isinstance(text, str) else ""


def grade_provider_status(case: GradedCase) -> Grade:
    """
    Check that the case ended the way the case set asked it to end.

    A provider that failed is the worst result a question can have, so it is
    reported as a failure of the case rather than as a case with nothing to
    say. Nothing here can turn a failure into a pass.
    """
    if case.expected_outcome is None:
        return Grade(
            "provider_status", UNKNOWN, "the case declares no expected outcome"
        )
    if case.outcome == case.expected_outcome:
        return Grade("provider_status", PASSED, case.outcome)
    return Grade(
        "provider_status",
        FAILED,
        f"the run recorded {case.outcome}, the case expected {case.expected_outcome}",
    )


def grade_required_abstention(case: GradedCase) -> Grade:
    """
    Check that the run refused the questions it had to refuse, and only those.

    A question the evidence does not reach has one honest answer, and a question
    the evidence does reach has a real one. Abstaining on the second is a defect
    too, so this grader is asked about every case rather than only about the ones
    that declare an abstention.
    """
    if case.requires_abstention is None:
        return Grade("required_abstention", UNKNOWN, "the case declares no abstention")
    if case.requires_abstention:
        if case.outcome == "abstained":
            return Grade(
                "required_abstention", PASSED, "abstained as the case required"
            )
        return Grade(
            "required_abstention",
            FAILED,
            f"the case required an abstention and the run recorded {case.outcome}",
        )
    if case.outcome == "abstained":
        return Grade(
            "required_abstention",
            FAILED,
            "the run abstained on a question the case says the evidence answers",
        )
    if case.outcome == "answered":
        return Grade("required_abstention", PASSED, "answered as the case required")
    return Grade(
        "required_abstention",
        FAILED,
        f"the run recorded {case.outcome}, which is not the abstention or the "
        "answer the case called for",
    )


def grade_exact_evidence(case: GradedCase) -> Grade:
    """
    Check that the answer's own citations reach the evidence that says it.

    The question is not whether the Passage was retrieved — retrieval is graded
    on its own — but whether the claims the model attached to its sentences name
    a Passage containing the expected wording. A right answer pointing at the
    wrong Passage is still a citation a reader cannot follow.
    """
    if not case.expected_evidence:
        return Grade(
            "exact_evidence", UNKNOWN, "the case declares no expected evidence"
        )
    if not case.claims:
        return Grade(
            "exact_evidence", UNKNOWN, "the answer declared no claims to check"
        )
    texts = case.source_texts
    cited = set(case.cited)
    missing = [
        snippet
        for snippet in case.expected_evidence
        if not any(
            contains_snippet(texts.get(source_id, ""), snippet) for source_id in cited
        )
    ]
    if not missing:
        return Grade("exact_evidence", PASSED, f"{len(case.expected_evidence)} found")
    return Grade(
        "exact_evidence", FAILED, "no cited passage holds: " + ", ".join(missing)
    )


def grade_claims_schema(case: GradedCase) -> Grade:
    """
    Check that the claims the run stored are claims the app can read back.

    The model writes the block, so what comes back is not guaranteed to be one.
    A claim with no text, a sources value that is not a list, or the same Passage
    cited twice is a claim the app could not show a reader as a citation.
    """
    if not case.claims:
        return Grade("claims_schema", UNKNOWN, "the answer declared no claims to read")
    problems: list[str] = []
    for position, claim in enumerate(case.claims, start=1):
        if not _claim_text(claim):
            problems.append(f"claim {position} has no text")
            continue
        if not isinstance(claim.get("sources"), list):
            problems.append(f"claim {position} has no list of sources")
            continue
        declared = [
            source_id
            for source_id in claim["sources"]
            if isinstance(source_id, str) and source_id.strip()
        ]
        if len(_declared_ids(claim)) != len(declared):
            problems.append(
                f"claim {position} cites a passage twice: {', '.join(declared)}"
            )
    if problems:
        return Grade("claims_schema", FAILED, "; ".join(problems))
    return Grade("claims_schema", PASSED, f"{len(case.claims)} claims readable")


def grade_source_ids(case: GradedCase) -> Grade:
    """
    Check that every claim points at a Passage the question was given.

    The answer path already rejects an id that names nothing supplied, so this
    grader is asking the narrower question: whether the record the reader gets
    back has a citation on each claim that can be followed to a Passage.
    """
    if not case.claims:
        return Grade("source_ids", UNKNOWN, "the answer declared no claims to check")
    supplied = set(case.source_texts)
    unsupported = [
        f"{_claim_text(claim) or f'claim {position}'} "
        f"(cited {', '.join(_declared_ids(claim)) or 'nothing'})"
        for position, claim in enumerate(case.claims, start=1)
        if not any(source_id in supplied for source_id in _declared_ids(claim))
    ]
    if unsupported:
        return Grade(
            "source_ids",
            FAILED,
            "claims citing no supplied passage: " + "; ".join(unsupported),
        )
    return Grade(
        "source_ids", PASSED, f"{len(case.claims)} claims cite supplied passages"
    )


def grade_page_references(case: GradedCase) -> Grade:
    """
    Check that a page the reader is sent to is the page the Passage came from.

    A page number is a promise about where to look, so it is checked against the
    citations on the sentence that made it. A claim that says "page 4" is
    checked against the Passages that claim cites, and a page in the answer's
    own prose is checked against every Passage the answer cited. A Passage with
    no page cannot keep either promise, and neither can a claim that cites it.
    """
    if not case.claims:
        return Grade(
            "page_references", UNKNOWN, "the answer declared no claims to check"
        )
    supplied = {str(source.get("source_id")): source for source in case.sources}
    if not any(_page(source) is not None for source in supplied.values()):
        return Grade(
            "page_references", UNKNOWN, "the Passages supplied carry no page numbers"
        )
    cited = [supplied[source_id] for source_id in case.cited if source_id in supplied]
    pageless = [
        str(source.get("source_id")) for source in cited if _page(source) is None
    ]
    if pageless:
        return Grade(
            "page_references",
            FAILED,
            "cited passage has no page: " + ", ".join(pageless),
        )
    cited_pages = {_page(source) for source in cited}
    wrong: list[str] = []
    for claim in case.claims:
        pages = {_page(supplied[source_id]) for source_id in _declared_ids(claim)}
        for page in sorted(pages_in(_claim_text(claim))):
            if page not in pages:
                wrong.append(
                    f"a claim citing {', '.join(_declared_ids(claim))} says page "
                    f"{page} and those Passages are on "
                    + ", ".join(str(number) for number in sorted(pages))
                )
    for page in sorted(pages_in(case.answer or "")):
        if page not in cited_pages:
            wrong.append(
                f"the answer says page {page} and no cited passage is on it; cited "
                "pages: " + ", ".join(str(number) for number in sorted(cited_pages))
            )
    if wrong:
        return Grade("page_references", FAILED, "; ".join(wrong))
    return Grade("page_references", PASSED, f"pages cited: {len(cited_pages)}")


def _page(source: dict[str, Any]) -> int | None:
    """Return the page a Passage came from, as an int, or None when it has none."""
    page = source.get("page")
    if isinstance(page, bool) or page is None:
        return None
    try:
        return int(page)
    except (TypeError, ValueError):
        return None


def citation_precision(case: GradedCase) -> float | None:
    """
    Return the share of an answer's citations that support the claim they sit on.

    A citation supports a claim when the Passage it names repeats the claim's
    own content words. That is a proxy, and it is a crude one: it cannot tell
    support from a shared topic, and it says nothing about a claim that is
    supported in a way it did not phrase. It is here because it is reproducible
    and it catches the failure that matters most — a claim pointed at a Passage
    about something else entirely.

    None when the answer made no citations, which is not a precision of zero.
    """
    texts = case.source_texts
    pairs = 0
    supporting = 0
    for claim in case.claims:
        words = content_words(_claim_text(claim))
        for source_id in _declared_ids(claim):
            pairs += 1
            if not words:
                continue
            passage = content_words(texts.get(source_id, ""))
            if len(words & passage) / len(words) >= SUPPORTING_OVERLAP:
                supporting += 1
    return supporting / pairs if pairs else None


def citation_recall(case: GradedCase) -> float | None:
    """
    Return the share of an answer's claims that cite a supplied Passage.

    A claim with an empty sources list is a claim the reader cannot check, so it
    is the missing citation this measures. None when the answer declared no
    claims, which is a different defect and is graded as one.
    """
    if not case.claims:
        return None
    supplied = set(case.source_texts)
    cited = sum(
        1
        for claim in case.claims
        if any(source_id in supplied for source_id in _declared_ids(claim))
    )
    return cited / len(case.claims)


#: The deterministic graders a run applies, in the order a report lists them.
DETERMINISTIC_GRADERS = (
    grade_provider_status,
    grade_required_abstention,
    grade_exact_evidence,
    grade_claims_schema,
    grade_source_ids,
    grade_page_references,
)


@dataclass
class GradedResult:
    """Every deterministic grade for one case, and the two citation fractions."""

    grades: dict[str, Grade] = field(default_factory=dict)
    citation_precision: float | None = None
    citation_recall: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the case's grades as a report stores them."""
        return {
            "grades": {name: grade.to_dict() for name, grade in self.grades.items()},
            "citation_precision": self.citation_precision,
            "citation_recall": self.citation_recall,
        }


def grade_case(case: GradedCase) -> GradedResult:
    """Run every deterministic grader over one case."""
    grades = [grader(case) for grader in DETERMINISTIC_GRADERS]
    return GradedResult(
        grades={grade.name: grade for grade in grades},
        citation_precision=citation_precision(case),
        citation_recall=citation_recall(case),
    )


def outcome_for_score(score: float | None) -> str:
    """Return the outcome a fraction earns: full marks pass, less than full fails."""
    if score is None:
        return UNKNOWN
    return PASSED if score >= 1.0 else FAILED
