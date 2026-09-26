"""
Running a labeled case set through the production answer path.

Every case is asked the way a reader asks it: the same document context, the
same retrieval, the same bounded prompt, the same citation validation, the same
abstention decision, and the same persistence. A case is therefore a question
about the application rather than a question about a reimplementation of it,
and what it ends as — an answer, an abstention, a provider failure, an
unusable citation, a failed save — is recorded as the outcome it was rather
than folded into a score.

The run record carries the index manifest and generation, the retrieval
method, the prompt version, the provider, the model, and the settings that
produced every answer, so two runs can be compared or a result can be
explained. Only an answer the model actually generated is handed to a grader:
retrieved context quoted back after a provider failure is a fallback, not an
answer, and scoring it would credit the model with the retrieval.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from evaluation import judge as judge_module
from evaluation.harness import EvaluationEnvironment
from evaluation.metrics import RetrievalReport, hit_at_k, recall_at_k, summarize
from services.answering import (
    AnswerEvent,
    AnswerRequest,
    AnswerSettings,
    Refusal,
)
from services.citations import PROMPT_VERSION
from services.llm.base import is_context_fallback

FIXTURE_PATH = Path(__file__).parent / "fixture.json"

DEFAULT_K = 5

#: A case that reached the model and was stored. Every other outcome is named
#: by the event that ended it, so there is one vocabulary rather than three.
ANSWERED = "answered"
ABSTAINED = "abstained"
PROVIDER_ERROR = "provider_error"
CITATION_ERROR = "citation_error"
PERSISTENCE_ERROR = "persistence_error"
CANCELLED = "cancelled"

#: The question was not asked at all: the Document could not be asked about.
REFUSED = "refused"
#: The stored text is a failed call quoting the evidence back, not an answer.
CONTEXT_FALLBACK = "context_fallback"

#: Every way a case can end. A run reports all of them, because a case that
#: abstained is a different result from a case that answered badly, and neither
#: is the same as a case that never reached the model.
OUTCOMES = (
    ANSWERED,
    ABSTAINED,
    PROVIDER_ERROR,
    CITATION_ERROR,
    PERSISTENCE_ERROR,
    CANCELLED,
    REFUSED,
    CONTEXT_FALLBACK,
)

#: The events that end a case, and the one whose name is not already an outcome.
_TERMINAL_EVENTS = (
    "done",
    ABSTAINED,
    PROVIDER_ERROR,
    CITATION_ERROR,
    PERSISTENCE_ERROR,
    CANCELLED,
)


def outcome_for(event_name: str) -> str:
    """Return the outcome a terminal event names."""
    return ANSWERED if event_name == "done" else event_name


def load_fixture(path: Path = FIXTURE_PATH) -> dict:
    """Read the labeled case set from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class CaseResult:
    """What one case did, and the evidence the run can measure it against."""

    id: str
    document: str
    question: str
    outcome: str
    detail: str | None = None
    turn_id: str | None = None
    retrieval: dict[str, Any] = field(default_factory=dict)
    retrieved_chunks: int = 0
    retrieved_texts: list[str] = field(default_factory=list)
    hit_at_k: bool | None = None
    recall_at_k: float | None = None
    answer: str | None = None
    claims: list[dict[str, Any]] = field(default_factory=list)
    grounded: bool | None = None
    truncated: bool | None = None
    finish_reason: str | None = None
    context: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    faithfulness: float | None = None
    verdict: str | None = None

    @property
    def generated(self) -> bool:
        """Report whether a model actually wrote the text a grader would read."""
        return (
            self.outcome == ANSWERED
            and self.answer is not None
            and not is_context_fallback(self.answer)
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the case as a report stores it."""
        result = asdict(self)
        result.pop("retrieved_texts")
        result["generated"] = self.generated
        return result


@dataclass
class EvaluationReport:
    """One run: what it measured, how it was configured, and every case."""

    run: dict[str, Any]
    retrieval: RetrievalReport
    outcomes: dict[str, int]
    faithfulness: dict[str, Any]
    cases: list[CaseResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return the report as the CLI's JSON output."""
        return {
            "run": self.run,
            "retrieval": asdict(self.retrieval),
            "outcomes": self.outcomes,
            "faithfulness": self.faithfulness,
            "cases": [case.as_dict() for case in self.cases],
        }


def _terminal(events: Sequence[AnswerEvent]) -> tuple[str, dict[str, Any]]:
    """Return the one event that decided a case, and its payload."""
    for name in reversed(_TERMINAL_EVENTS):
        for event in reversed(events):
            if event.name == name:
                return event.name, dict(event.payload)
    raise ValueError("the answer path ended without an outcome")


def run_case(
    case: dict,
    environment: EvaluationEnvironment,
    model,
    api_key: str = "evaluation",
    *,
    k: int = DEFAULT_K,
) -> CaseResult:
    """
    Ask one case through the answer path and report what happened.

    A multi-turn case commits its earlier questions to the Document's
    Conversation first, so the answer path reads recorded Turns rather than
    being handed a transcript.
    """
    environment.seed_turns(case["document"], case.get("follow_up", []))
    request = AnswerRequest(
        filename=environment.stored_name(case["document"]),
        query=case["question"],
        provider=environment.provider(model, api_key),
        model=model,
    )
    resolved = environment.answer_service.resolve(request)
    if isinstance(resolved, Refusal):
        # Retrieval never ran, so there is nothing to score: the case is
        # reported as refused rather than as a retrieval that found nothing.
        return CaseResult(
            id=case["id"],
            document=case["document"],
            question=case["question"],
            outcome=REFUSED,
            detail=resolved.category,
        )

    events = list(environment.answer_service.stream(resolved))
    name, payload = _terminal(events)
    outcome = outcome_for(name)
    texts = [source.get("content") or "" for source in resolved.sources]
    result = CaseResult(
        id=case["id"],
        document=case["document"],
        question=case["question"],
        outcome=outcome,
        detail=payload.get("category") or payload.get("reason"),
        turn_id=resolved.turn_id,
        retrieval=resolved.retrieval.payload,
        retrieved_chunks=len(texts),
        retrieved_texts=texts,
        hit_at_k=hit_at_k(texts, case.get("gold_snippets", []), k),
        recall_at_k=recall_at_k(texts, case.get("gold_snippets", []), k),
        context=resolved.context,
        provenance=resolved.provenance.to_dict(),
    )
    if outcome != ANSWERED:
        return result
    return _with_answer(result, payload)


def _with_answer(result: CaseResult, payload: dict[str, Any]) -> CaseResult:
    """Attach the stored answer, its claims, and how the model ended it."""
    answer = payload.get("answer")
    if is_context_fallback(answer):
        # A provider that failed its one-shot repair quoted the evidence back.
        # That text is not an answer, so it is reported as the fallback it is.
        return replace(result, outcome=CONTEXT_FALLBACK)
    return replace(
        result,
        answer=answer,
        claims=payload.get("claims") or [],
        grounded=payload.get("grounded"),
        truncated=payload.get("truncated"),
        finish_reason=payload.get("finish_reason"),
    )


def _score_faithfulness(case: CaseResult, judge_fn: Callable[[str], str]) -> CaseResult:
    """
    Grade one case's answer, and only when a model wrote it.

    Abstentions, provider failures, and refused cases have no answer to grade,
    and a fallback quote is document text rather than a model answer, so none of
    them reach the judge.
    """
    if not case.generated:
        return case
    answer = case.answer or ""
    context = "\n\n".join(case.retrieved_texts)
    verdict, score = judge_module.judge_faithfulness(
        case.question, answer, context, judge_fn
    )
    return replace(case, verdict=verdict, faithfulness=score)


def _run_record(
    environment: EvaluationEnvironment, model, k: int, cases: Sequence[CaseResult]
) -> dict[str, Any]:
    """
    Describe the run as a whole: what was indexed, and how it was configured.

    The index manifest and generation of every Document are recorded alongside
    the prompt version, provider, model, and settings, because a number is
    only comparable to another number when those match. The prompt version and
    the settings come from the configuration rather than from a case, so a run
    in which every case failed still says what it was measuring.
    """
    retrieved = [case for case in cases if case.provenance]
    return {
        "k": k,
        "prompt_version": PROMPT_VERSION,
        "provider": model.provider,
        "model": model.id,
        "settings": AnswerSettings.from_settings(environment.settings).to_dict(),
        "documents": {
            name: environment.index_state(name)
            for name in sorted(environment.documents)
        },
        "retrieval_methods": sorted(
            {case.provenance["retrieval_method"] for case in retrieved}
        ),
    }


def evaluate(
    fixture: dict,
    environment: EvaluationEnvironment,
    model,
    api_key: str = "evaluation",
    judge_fn: Callable[[str], str] | None = None,
    k: int = DEFAULT_K,
) -> EvaluationReport:
    """
    Run every case in the fixture through the answer path.

    ``judge_fn`` may be None to skip faithfulness scoring, which is what a
    retrieval-only run does: the case outcomes and hit/recall still describe
    the production path.
    """
    cases: list[CaseResult] = []
    for case in fixture["questions"]:
        result = run_case(case, environment, model, api_key, k=k)
        if judge_fn is not None:
            result = _score_faithfulness(result, judge_fn)
        cases.append(result)

    retrieval = summarize(
        [case.as_dict() for case in cases if case.hit_at_k is not None], k
    )
    scores = [case.faithfulness for case in cases if case.faithfulness is not None]
    return EvaluationReport(
        run=_run_record(environment, model, k, cases),
        retrieval=retrieval,
        outcomes={
            outcome: sum(1 for case in cases if case.outcome == outcome)
            for outcome in OUTCOMES
        },
        faithfulness={
            "mean": sum(scores) / len(scores) if scores else None,
            "judged": len(scores),
            "faithful": sum(1 for score in scores if score == 1.0),
        },
        cases=cases,
    )
