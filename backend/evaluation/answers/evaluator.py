"""
Running a labeled case set through the production answer path, and measuring it.

Every case is asked the way a reader asks it: the same document context, the
same retrieval, the same bounded prompt, the same citation validation, the same
abstention decision, and the same persistence. A case is therefore a question
about the application rather than a question about a reimplementation of it,
and what it ends as — an answer, an abstention, a provider failure, an
unusable citation, a failed save — is recorded as the outcome it was rather
than folded into a score.

The run record carries the index manifest and generation, the retrieval
method, the prompt version, the provider, the model, the settings that produced
every answer, and the identity and rubric version of the judge, so two runs can
be compared or a result can be explained.

Measurement is split so that a failure cannot flatter a number. Only text a
model wrote is handed to a grader, so a provider that died is reported as a
failure rather than as an answer with no good citations. Latency is split into
the run's first case and the rest, because the first one pays for loading
whatever loads lazily and nobody's second question does. Cost is computed only
for the cases that reached a model, and from the token counts the run measured.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from evaluation.answers import judge as judge_module
from evaluation.answers.calibration import load_calibration, run_calibration
from evaluation.answers.dataset import REPORTED, Case, Dataset, abstention_required
from evaluation.answers.graders import (
    UNKNOWN,
    Grade,
    GradedCase,
    grade_case,
    outcome_for_score,
)
from evaluation.answers.harness import EvaluationEnvironment
from evaluation.answers.judge import Judge
from evaluation.answers.metrics import (
    LatencySummary,
    MetricSummary,
    PricedModel,
    RetrievalReport,
    cost_for,
    latency_summary,
    retrieval_scores,
    summarize,
)
from evaluation.answers.outcomes import (
    ABSTAINED,
    ANSWERED,
    CANCELLED,
    CITATION_ERROR,
    CONTEXT_FALLBACK,
    OUTCOMES,
    PERSISTENCE_ERROR,
    PROVIDER_ERROR,
    REFUSED,
)
from services.answering import (
    AnswerRequest,
    AnswerSettings,
    ResolvedTurn,
)
from services.streaming import AnswerEvent, Refusal
from services.text import token_len
from services.citations import PROMPT_VERSION, claims_block
from services.llm.base import is_context_fallback
from services.prompts import SYSTEM_INSTRUCTION

DEFAULT_K = 5

#: The events that end a case, and the one whose name is not already an outcome.
_TERMINAL_EVENTS = (
    "done",
    ABSTAINED,
    PROVIDER_ERROR,
    CITATION_ERROR,
    PERSISTENCE_ERROR,
    CANCELLED,
)

#: A case's phase. The first case of a run that reached retrieval is where
#: anything loaded lazily is loaded, so its seconds are reported apart from the
#: rest: a p95 that averaged model loading in would describe a cost the reader
#: never pays twice.
COLD_START = "cold_start"
WARM = "warm"
#: The case never reached retrieval, so it has no seconds to report either way.
NOT_MEASURED = "not_measured"


def outcome_for(event_name: str) -> str:
    """Return the outcome a terminal event names."""
    return ANSWERED if event_name == "done" else event_name


@dataclass
class CaseResult:
    """What one case did, and the evidence the run can measure it against."""

    id: str
    document: str
    question: str
    outcome: str
    split: str
    category: str
    expected_outcome: str
    detail: str | None = None
    turn_id: str | None = None
    retrieval: dict[str, Any] = field(default_factory=dict)
    retrieved_chunks: int = 0
    retrieved_texts: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    hit_at_k: bool | None = None
    recall_at_k: float | None = None
    reciprocal_rank: float | None = None
    ndcg_at_k: float | None = None
    answer: str | None = None
    claims: list[dict[str, Any]] = field(default_factory=list)
    grounded: bool | None = None
    truncated: bool | None = None
    finish_reason: str | None = None
    context: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    faithfulness: float | None = None
    verdict: str | None = None
    correctness: float | None = None
    correctness_verdict: str | None = None
    expected_answer: str | None = None
    expected_evidence: list[str] = field(default_factory=list)
    #: What a person checking this answer has to find, in the reviewer's words.
    #: It reaches the judge, because a reference answer is one wording of what
    #: counts as correct and not the whole of it.
    rubric: str = ""
    requires_abstention: bool | None = None
    grades: dict[str, Grade] = field(default_factory=dict)
    citation_precision: float | None = None
    citation_recall: float | None = None
    #: Whether these seconds include a lazy load. Set by the run, not the case:
    #: a case cannot know whether it is the first one asked.
    phase: str = NOT_MEASURED
    retrieval_seconds: float | None = None
    first_token_seconds: float | None = None
    total_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None

    @property
    def generated(self) -> bool:
        """Report whether a model actually wrote the text a grader would read."""
        return (
            self.outcome == ANSWERED
            and self.answer is not None
            and not is_context_fallback(self.answer)
        )

    def scored(self) -> GradedCase:
        """Return the case as the deterministic graders read it."""
        return GradedCase(
            id=self.id,
            outcome=self.outcome,
            answer=self.answer,
            claims=tuple(self.claims),
            sources=tuple(self.sources),
            expected_answer=self.expected_answer,
            expected_evidence=tuple(self.expected_evidence),
            requires_abstention=self.requires_abstention,
            expected_outcome=self.expected_outcome,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the case as a report stores it."""
        result = asdict(self)
        # A report is read, not re-ingested: it carries what was measured about
        # the evidence — every retrieval score, and the page each Passage came
        # from — rather than the Passage text the model read, which would make
        # every case the size of the context that was sent to a paid model.
        result.pop("retrieved_texts")
        sources = result.pop("sources")
        result["source_pages"] = {
            str(source.get("source_id")): source.get("page") for source in sources
        }
        result["generated"] = self.generated
        return result


@dataclass
class AnswerReport:
    """Every metric about the answers themselves, in one comparable shape."""

    correctness: MetricSummary
    faithfulness: MetricSummary
    citation_precision: MetricSummary
    citation_recall: MetricSummary
    abstention: MetricSummary
    finish_reasons: dict[str, int]
    generated: int
    truncated: int

    def to_dict(self) -> dict[str, Any]:
        """Return the answer report as a run record stores it."""
        return {
            "correctness": self.correctness.to_dict(),
            "faithfulness": self.faithfulness.to_dict(),
            "citation_precision": self.citation_precision.to_dict(),
            "citation_recall": self.citation_recall.to_dict(),
            "abstention": self.abstention.to_dict(),
            "generated": self.generated,
            "truncated": self.truncated,
            "finish_reasons": self.finish_reasons,
        }


@dataclass
class LatencyReport:
    """
    How long the run took, with the first case held apart from the rest.

    Every measurement is a pair: what the first question cost, and what the
    others cost. Reporting only the pair's mean, or only the tail, hides either
    the cost of a cold start or the cost a reader actually waits.
    """

    cold_start: dict[str, LatencySummary]
    steady_state: dict[str, LatencySummary]

    def to_dict(self) -> dict[str, Any]:
        """Return the latency report as a run record stores it."""
        return {
            "cold_start": {
                name: summary.to_dict() for name, summary in self.cold_start.items()
            },
            "steady_state": {
                name: summary.to_dict() for name, summary in self.steady_state.items()
            },
        }


@dataclass
class CostReport:
    """What the run's calls cost, over the cases that reached a model."""

    input_tokens: int
    output_tokens: int
    usd: float
    priced_cases: int
    model: str
    #: The prices the estimate used, so a reader can check it against the
    #: provider's own published table.
    input_cost_per_million_usd: float
    output_cost_per_million_usd: float

    def to_dict(self) -> dict[str, Any]:
        """Return the cost report as a run record stores it."""
        return asdict(self)


@dataclass
class EvaluationReport:
    """One run: what it measured, how it was configured, and every case."""

    run: dict[str, Any]
    retrieval: RetrievalReport
    outcomes: dict[str, int]
    answers: AnswerReport
    latency: LatencyReport
    cost: CostReport
    cases: list[CaseResult] = field(default_factory=list)
    calibration: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the report as the CLI's JSON output."""
        return {
            "run": self.run,
            "retrieval": asdict(self.retrieval),
            "outcomes": self.outcomes,
            "answers": self.answers.to_dict(),
            "latency": self.latency.to_dict(),
            "cost": self.cost.to_dict(),
            "calibration": self.calibration,
            "cases": [case.as_dict() for case in self.cases],
        }


def input_tokens_of(resolved: ResolvedTurn) -> int:
    """
    Return what one call costs the reader in input tokens.

    The measure lives in the telemetry module, where the answer path uses the
    same one for its traces: a run and a trace cannot disagree about what a
    call read.
    """
    from services.telemetry.usage import input_tokens

    return input_tokens(resolved.request.query, resolved.context, resolved.prior_turns)


def _terminal(events: Sequence[AnswerEvent]) -> tuple[str, dict[str, Any]]:
    """Return the one event that decided a case, and its payload."""
    for name in reversed(_TERMINAL_EVENTS):
        for event in reversed(events):
            if event.name == name:
                return event.name, dict(event.payload)
    raise ValueError("the answer path ended without an outcome")


def _streamed(environment: EvaluationEnvironment, resolved: ResolvedTurn):
    """
    Walk the answer path's events and read the timings it measured.

    The answer path already reads the clock around the stream, so a run reads
    those measurements rather than starting a second pair of its own: one
    request is timed once, and the report and the trace cannot disagree about
    how long it took.
    """
    events = list(environment.answer_service.stream(resolved))
    return (
        events,
        resolved.trace.first_token_seconds,
        round(resolved.trace.elapsed(), 6),
    )


def run_case(
    case: Case,
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
    environment.seed_turns(case.document, case.follow_up)
    request = AnswerRequest(
        filename=environment.stored_name(case.document),
        query=case.question,
        provider=environment.provider(model, api_key),
        model=model,
    )
    resolved = environment.answer_service.resolve(request)
    if isinstance(resolved, Refusal):
        # Retrieval never ran, so there is nothing to score: the case is
        # reported as refused rather than as a retrieval that found nothing.
        return _graded(
            CaseResult(
                id=case.id,
                document=case.document,
                question=case.question,
                outcome=REFUSED,
                split=case.split,
                category=case.category,
                expected_outcome=case.expected_outcome,
                detail=resolved.category,
            )
        )

    sources = list(resolved.sources)
    texts = [str(source.get("content") or "") for source in sources]
    scores = retrieval_scores(texts, case.expected_evidence, k)
    events, first_token, total = _streamed(environment, resolved)
    name, payload = _terminal(events)
    outcome = outcome_for(name)
    result = CaseResult(
        id=case.id,
        document=case.document,
        question=case.question,
        outcome=outcome,
        split=case.split,
        category=case.category,
        expected_outcome=case.expected_outcome,
        detail=payload.get("category") or payload.get("reason"),
        turn_id=resolved.turn_id,
        retrieval=resolved.retrieval.payload,
        retrieved_chunks=len(texts),
        retrieved_texts=texts,
        sources=sources,
        context=resolved.context,
        provenance=resolved.provenance.to_dict(),
        expected_answer=case.expected_answer,
        expected_evidence=list(case.expected_evidence),
        rubric=case.rubric,
        requires_abstention=abstention_required(case),
        retrieval_seconds=resolved.provenance.retrieval_seconds,
        first_token_seconds=first_token,
        total_seconds=total,
        # Every call carries the system instruction, whether or not this case
        # reaches the model, so a run's input total is the question, the
        # transcript, the evidence, and the instruction the provider is billed
        # for reading on all of them.
        input_tokens=input_tokens_of(resolved),
        **scores,
    )
    if outcome != ANSWERED:
        return _graded(result)
    return _graded(_with_answer(result, payload))


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
        # What the provider was asked to produce is the prose and the claims
        # block together, and it billed for both. The stored answer is the prose
        # alone, so the block is put back to count what was actually written.
        output_tokens=token_len(
            (answer or "") + claims_block(payload.get("claims") or [])
        ),
    )


def _graded(result: CaseResult) -> CaseResult:
    """Run the deterministic graders and read their verdicts back onto the case."""
    graded = grade_case(result.scored())
    return replace(
        result,
        grades=graded.grades,
        citation_precision=graded.citation_precision,
        citation_recall=graded.citation_recall,
    )


def _judge(case: CaseResult, judge: Judge) -> CaseResult:
    """
    Grade one case's answer with the model judge, and only a written one.

    Abstentions, provider failures, and refused cases have no answer to grade,
    and a fallback quote is document text rather than a model answer, so none of
    them reach the judge. Correctness is skipped when the case set declares no
    expected answer: there is nothing to compare the answer against.
    """
    if not case.generated:
        return case
    answer = case.answer or ""
    context = "\n\n".join(case.retrieved_texts)
    verdict, score = judge_module.judge_faithfulness(
        case.question, answer, context, judge
    )
    correctness_verdict, correctness = (None, None)
    if case.expected_answer:
        correctness_verdict, correctness = judge_module.judge_correctness(
            case.question, case.expected_answer, answer, context, judge, case.rubric
        )
    return replace(
        case,
        verdict=verdict,
        faithfulness=score,
        correctness_verdict=correctness_verdict,
        correctness=correctness,
    )


def _grade_scores(grades: Sequence[Grade]) -> tuple[list[float | None], list[str]]:
    """Return the scores and outcomes one deterministic metric summarizes from."""
    return (
        [
            None if grade.outcome == UNKNOWN else (1.0 if grade.passed else 0.0)
            for grade in grades
        ],
        [grade.outcome for grade in grades],
    )


def _by_outcome(cases: Sequence[CaseResult], name: str) -> list[Grade]:
    """Return one grader's verdicts across the run, in the order it asked."""
    return [case.grades[name] for case in cases if name in case.grades]


def _from_scores(scores: Sequence[float | None]) -> MetricSummary:
    """
    Summarize a metric whose cases each carry one number.

    A case with no number is one the grader could not decide: it is counted as
    unknown and left out of the mean, and it is never read as a zero.
    """
    return MetricSummary.of(scores, [outcome_for_score(score) for score in scores])


def _answer_report(cases: Sequence[CaseResult]) -> AnswerReport:
    """Summarize every metric about the answers a run produced."""
    correctness_scores = [case.correctness for case in cases]
    faithfulness_scores = [case.faithfulness for case in cases]
    precision_scores = [case.citation_precision for case in cases]
    recall_scores = [case.citation_recall for case in cases]
    abstention_scores, abstention_outcomes = _grade_scores(
        _by_outcome(cases, "required_abstention")
    )
    answered = [case for case in cases if case.outcome == ANSWERED]
    reasons: dict[str, int] = {}
    for case in answered:
        reason = case.finish_reason or "none"
        reasons[reason] = reasons.get(reason, 0) + 1
    return AnswerReport(
        correctness=_from_scores(correctness_scores),
        faithfulness=_from_scores(faithfulness_scores),
        citation_precision=_from_scores(precision_scores),
        citation_recall=_from_scores(recall_scores),
        abstention=MetricSummary.of(abstention_scores, abstention_outcomes),
        finish_reasons=reasons,
        generated=sum(1 for case in cases if case.generated),
        truncated=sum(1 for case in answered if case.truncated),
    )


#: The three things a case is timed on, in the order a report lists them.
MEASURED = ("retrieval_seconds", "first_token_seconds", "total_seconds")


def _samples(cases: Sequence[CaseResult], phase: str, name: str) -> list[float]:
    """Return the values one measurement holds for the cases in one phase."""
    return [
        value
        for case in cases
        if case.phase == phase
        for value in [getattr(case, name)]
        if value is not None
    ]


def _measure_latency(
    cases: Sequence[CaseResult], model: PricedModel
) -> tuple[LatencyReport, CostReport]:
    """Split the run's timings and its cost into the parts a reader compares."""
    # Only the cases that reached a model and wrote something are priced, and
    # the token totals count those same cases. A token total that included an
    # abstention's prompt would sit beside a dollar figure that excluded it.
    priced = [case for case in cases if case.cost_usd is not None]
    return (
        LatencyReport(
            cold_start={
                name: latency_summary(_samples(cases, COLD_START, name))
                for name in MEASURED
            },
            steady_state={
                name: latency_summary(_samples(cases, WARM, name)) for name in MEASURED
            },
        ),
        CostReport(
            input_tokens=sum(case.input_tokens or 0 for case in priced),
            output_tokens=sum(case.output_tokens or 0 for case in priced),
            usd=round(sum(case.cost_usd or 0.0 for case in priced), 6),
            priced_cases=len(priced),
            model=model.id,
            input_cost_per_million_usd=model.input_cost_per_million_usd,
            output_cost_per_million_usd=model.output_cost_per_million_usd,
        ),
    )


def _phase(result: CaseResult, asked: Sequence[CaseResult]) -> CaseResult:
    """
    Label a case as the run's cold start or as steady state.

    The cold start is the first case whose retrieval actually ran, not the first
    case in the fixture. A case that was refused before retrieval — a Document
    that could not be asked about, a question the app will not accept — measures
    nothing, and calling the case after it warm would put the seconds that paid
    for a lazy load into the steady-state distribution.
    """
    if result.retrieval_seconds is None:
        return replace(result, phase=NOT_MEASURED)
    already_cold = any(case.phase == COLD_START for case in asked)
    return replace(result, phase=WARM if already_cold else COLD_START)


def _run_record(
    environment: EvaluationEnvironment,
    model,
    k: int,
    cases: Sequence[CaseResult],
    judge: Judge | None,
    dataset: Dataset,
    split: str,
    held_back: Sequence[str],
) -> dict[str, Any]:
    """
    Describe the run as a whole: what it measured, and how it was configured.

    The version of the case set and the split that was run are recorded beside
    the index manifest and generation of every Document, the prompt version, the
    provider, the model, and the settings, because a number is only comparable
    to another number when those match. The set and the configuration come from
    outside the run rather than from a case, so a run in which every case failed
    still says what it was measuring.
    """
    retrieved = [case for case in cases if case.provenance]
    return {
        "k": k,
        "dataset": dataset.version,
        "split": split,
        "cases": len(cases),
        "held_back": list(held_back),
        "prompt_version": PROMPT_VERSION,
        "provider": model.provider,
        "model": model.id,
        "judge": judge.to_dict() if judge else None,
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
    dataset: Dataset,
    environment: EvaluationEnvironment,
    model,
    api_key: str = "evaluation",
    judge: Judge | None = None,
    k: int = DEFAULT_K,
    calibrate: bool = True,
    split: str = REPORTED,
    include_faults: bool = False,
) -> EvaluationReport:
    """
    Run one split of the case set through the answer path and measure the run.

    The reported split is the default because it is the one a result is quoted
    from. The cases carrying a fault are held back unless ``include_faults``
    asks for them, because their expected outcome is only reachable in a run
    that injects the fault, and a reported number that counted a deliberately
    broken provider would be reporting the test. The cases left behind are
    named in the run record rather than dropped quietly.

    ``judge`` may be None to skip the two model-graded metrics and the
    calibration set, which is what a retrieval-only run does: the deterministic
    graders, the case outcomes, and the retrieval scores still describe the
    production path, and nothing is reported as judged that was not judged.
    """
    asked = dataset.cases_for(split, include_faults=include_faults)
    held_back = (
        [] if include_faults else [case.id for case in dataset.faults_for(split)]
    )
    cases: list[CaseResult] = []
    for case in asked:
        result = run_case(case, environment, model, api_key, k=k)
        if judge is not None:
            result = _judge(result, judge)
        if result.generated:
            result = replace(
                result,
                cost_usd=cost_for(result.input_tokens, result.output_tokens, model),
            )
        cases.append(_phase(result, cases))

    retrieval = summarize(
        [case.as_dict() for case in cases if case.hit_at_k is not None],
        k,
        questions=[case.id for case in cases if case.hit_at_k is not None],
    )
    latency, cost = _measure_latency(cases, model)
    calibration = (
        run_calibration(load_calibration(), judge).to_dict()
        if judge is not None and calibrate
        else None
    )
    return EvaluationReport(
        run=_run_record(environment, model, k, cases, judge, dataset, split, held_back),
        retrieval=retrieval,
        outcomes={
            outcome: sum(1 for case in cases if case.outcome == outcome)
            for outcome in OUTCOMES
        },
        answers=_answer_report(cases),
        latency=latency,
        cost=cost,
        calibration=calibration,
        cases=cases,
    )
