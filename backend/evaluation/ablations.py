"""
Measuring one configuration against the same labeled questions.

An ablation is a question about the application, asked of the application: the
same documents, indexed the same way, retrieved through the same
:class:`~services.retrieval.vector_service.VectorService`, scored with the same
four retrieval numbers the answer-path report uses. What changes between two
results is the one thing the experiment declared, so a difference between the
two columns belongs to that thing and not to the hour, the machine, or a
different question.

Two kinds of knob appear here and they cost different things. A retrieval knob
— the method, how many candidates are fetched, whether the reranker runs for
this call, whether the question is expanded — is an argument to one call, so it
is measured against the index that already exists. A chunking knob changes the
vectors themselves, so the documents are indexed again and the seconds of that
indexing are part of what the variant costs.

A retrieval-only measurement says so. It reports retrieval quality, what
retrieval cost in seconds, and how many candidates came back. It reports no
answer, citation, abstention, token, or dollar figure, because nothing asked a
model anything, and a report that left those fields empty would read as a run in
which they were measured and came to zero.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from evaluation.dataset import Case, Dataset
from evaluation.experiments import Experiment
from evaluation.harness import EvaluationEnvironment
from evaluation.metrics import (
    RetrievalReport,
    latency_summary,
    retrieval_scores,
    summarize,
)
from evaluation.report import evidence_for_retrieval
from services.retrieval.query_expansion import expand_query
from services.retrieval.vector_service import FETCH_K

#: The retrieval number a comparison reads. Hit rate alone cannot tell a
#: variant that found the passage first from one that found it fifth.
RANKING = "ndcg_at_k"


@dataclass
class CaseRetrieval:
    """What one question retrieved, and how long that took."""

    id: str
    document: str
    category: str
    outcome: str
    method: str
    candidates: int
    seconds: float
    hit_at_k: bool
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float

    def to_dict(self) -> dict[str, Any]:
        """Return the question's row as a result stores it."""
        return {
            "id": self.id,
            "document": self.document,
            "category": self.category,
            "outcome": self.outcome,
            "method": self.method,
            "candidates": self.candidates,
            "seconds": self.seconds,
            "hit_at_k": self.hit_at_k,
            "recall_at_k": self.recall_at_k,
            "reciprocal_rank": self.reciprocal_rank,
            "ndcg_at_k": self.ndcg_at_k,
        }


@dataclass
class AblationResult:
    """One experiment's retrieval numbers, its cost in seconds, and its rows."""

    experiment: str
    family: str
    retrieval: RetrievalReport
    latency: dict[str, Any]
    #: The version of the case set the questions came from, and the split of it
    #: that was asked, so two results are only compared when they were asked the
    #: same questions of the same set.
    dataset: str = ""
    split: str = ""
    #: What each document's index was, and the manifest it was built under. A
    #: chunking variant indexed the documents again, so its index is not the
    #: baseline's and a report has to carry both.
    index: dict[str, Any] = field(default_factory=dict)
    #: The cases of this split a reported run leaves out, named so that a
    #: reader can see which failures the report does not cover.
    held_back: list[str] = field(default_factory=list)
    cases: list[CaseRetrieval] = field(default_factory=list)
    #: What this result measured, and what it did not. A report reads this
    #: rather than inferring from absent fields.
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return the result as a report stores it."""
        return {
            "experiment": self.experiment,
            "family": self.family,
            "dataset": self.dataset,
            "split": self.split,
            "held_back": self.held_back,
            "index": self.index,
            "retrieval": {
                "questions": self.retrieval.questions,
                "k": self.retrieval.k,
                "hit_rate": self.retrieval.hit_rate,
                "recall": self.retrieval.recall,
                "mrr": self.retrieval.mrr,
                "ndcg": self.retrieval.ndcg,
                "per_question": [row for row in self.retrieval.per_question],
            },
            "latency": {
                name: summary.to_dict() for name, summary in self.latency.items()
            },
            "evidence": self.evidence,
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass
class Comparison:
    """How one experiment moved against the baseline, and on which questions."""

    baseline: str
    variant: str
    delta: dict[str, float]
    latency_delta: dict[str, dict[str, float | None]]
    regressions: list[dict[str, Any]] = field(default_factory=list)
    improvements: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return the comparison as a report stores it."""
        return {
            "baseline": self.baseline,
            "variant": self.variant,
            "delta": self.delta,
            "latency_delta": self.latency_delta,
            "regressions": self.regressions,
            "improvements": self.improvements,
        }


def _question_for(
    environment: EvaluationEnvironment, case: Case, experiment: Experiment
) -> str:
    """
    Return the text retrieval is asked with, for one experiment.

    Expansion is the production behaviour unless the experiment turns it off, so
    the baseline measures what a reader's follow-up question actually retrieves.
    A question that names its subject is returned unchanged either way: the
    expansion itself decides that.
    """
    if experiment.query_expansion == "none":
        return case.question
    budget = environment.settings.query_context.max_expansion_chars
    return expand_query(case.question, case.follow_up, max_chars=budget).expanded_query


def _rerank_for(environment: EvaluationEnvironment, experiment: Experiment) -> bool:
    """
    Return whether the reranker runs for this experiment.

    The baseline reports the gate the application ships. An experiment that sets
    the knob says so outright, so the reranker's own gate is not consulted and
    a column cannot depend on a machine's setting.
    """
    if experiment.rerank is None:
        return environment.settings.rerank.enabled
    return experiment.rerank


def _one_case(
    environment: EvaluationEnvironment,
    case: Case,
    experiment: Experiment,
    k: int,
) -> CaseRetrieval:
    """Retrieve for one case through the production service and score it."""
    stored = environment.stored_name(case.document)
    record = environment.repositories.files.get_file(stored) or {}
    question = _question_for(environment, case, experiment)
    embedding = environment.embedding_service.embed_texts(question)[0]
    started = environment.clock()
    result = environment.vector_service.query_vectors(
        embedding,
        stored,
        top_k=experiment.candidate_depth or FETCH_K,
        query_text=question,
        rerank=_rerank_for(environment, experiment),
        method=experiment.method,
        generation=record.get("index_generation"),
        include_legacy=record.get("index_generation") is None,
    )
    seconds = environment.clock() - started
    texts = [str(source.get("content") or "") for source in result.sources]
    return CaseRetrieval(
        id=case.id,
        document=case.document,
        category=case.category,
        outcome=result.outcome,
        method=result.method,
        candidates=len(texts),
        seconds=seconds,
        **retrieval_scores(texts, case.expected_evidence, k),
    )


def measure(
    environment: EvaluationEnvironment,
    dataset: Dataset,
    cases: Sequence[Case],
    experiment: Experiment,
    k: int = 5,
) -> AblationResult:
    """
    Measure one experiment on the given cases of the set.

    ``dataset`` is the whole set even when ``cases`` is a handful of it, because
    a result that names which version of the set its questions came from is
    comparable to another result and one that does not is an anecdote.
    """
    asked = [_one_case(environment, case, experiment, k) for case in cases]
    splits = {case.split for case in cases}
    split = splits.pop() if len(splits) == 1 else ""
    retrieval = summarize(
        [case.to_dict() for case in asked],
        k,
        questions=[case.id for case in asked],
    )
    return AblationResult(
        experiment=experiment.id,
        family=experiment.family,
        retrieval=retrieval,
        latency={
            "retrieval_seconds": latency_summary([case.seconds for case in asked]),
        },
        cases=asked,
        split=split,
        index={
            name: environment.index_state(name)
            for name in sorted(environment.documents)
        },
        evidence=evidence_for_retrieval(len(asked)),
        dataset=dataset.version,
        # The cases that only mean anything with a fault injected are left out
        # of a reported split, and named here rather than dropped quietly.
        held_back=[case.id for case in dataset.faults_for(split)],
    )


def _rows(result: AblationResult) -> dict[str, dict[str, Any]]:
    """Return one result's questions by id, with the numbers a move is read on."""
    return {
        case.id: {
            "hit_at_k": case.hit_at_k,
            "recall_at_k": case.recall_at_k,
            "reciprocal_rank": case.reciprocal_rank,
            "ndcg_at_k": case.ndcg_at_k,
        }
        for case in result.cases
    }


def _move(
    baseline: dict[str, Any], variant: dict[str, Any], name: str
) -> dict[str, Any]:
    """Return one question's before and after, with what changed between them."""
    return {
        "id": name,
        "baseline": baseline,
        "variant": variant,
        "delta": {
            metric: round(variant[metric] - baseline[metric], 6)
            for metric in ("hit_at_k", "recall_at_k", "reciprocal_rank", "ndcg_at_k")
        },
    }


def compare(baseline: AblationResult, variant: AblationResult) -> Comparison:
    """
    Return how one experiment moved against the baseline, question by question.

    A move is a change in the ranking number, so a variant that reordered the
    candidates without changing whether the answer was found is not reported as
    a regression, and one that found it first is. The questions behind each move
    are named, because a mean that moved is a question, not an answer.
    """
    before, after = _rows(baseline), _rows(variant)
    common = [name for name in before if name in after]
    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    for name in common:
        if after[name][RANKING] < before[name][RANKING]:
            regressions.append(_move(before[name], after[name], name))
        elif after[name][RANKING] > before[name][RANKING]:
            improvements.append(_move(before[name], after[name], name))
    return Comparison(
        baseline=baseline.experiment,
        variant=variant.experiment,
        delta={
            "hit_rate": variant.retrieval.hit_rate - baseline.retrieval.hit_rate,
            "recall": variant.retrieval.recall - baseline.retrieval.recall,
            "mrr": variant.retrieval.mrr - baseline.retrieval.mrr,
            "ndcg": variant.retrieval.ndcg - baseline.retrieval.ndcg,
        },
        latency_delta={
            name: {
                quantile: _difference(summary, variant.latency[name], quantile)
                for quantile in ("p50", "p95")
            }
            for name, summary in baseline.latency.items()
            if name in variant.latency
        },
        regressions=regressions,
        improvements=improvements,
    )


def _difference(before, after, quantile: str) -> float | None:
    """Return one quantile's change, or None when either side has no samples."""
    first, second = getattr(before, quantile), getattr(after, quantile)
    if first is None or second is None:
        return None
    return second - first


def run(
    experiments: Sequence[Experiment],
    cases: Sequence[Case],
    environment_for: Callable[[Experiment], EvaluationEnvironment],
    *,
    dataset: Dataset,
    k: int = 5,
    close: Callable[[EvaluationEnvironment], None] | None = None,
) -> tuple[AblationResult, ...]:
    """
    Measure a set of experiments, rebuilding the environment only when needed.

    ``environment_for`` builds the application an experiment is measured in,
    which lets a caller index real documents for the chunking experiments and
    reuse one index for everything else. ``close`` releases an environment the
    next experiment cannot use, and the last one when the run is over, so a
    reindexed set of vectors and its Documents are cleaned up rather than left
    behind in the store.
    """
    results: list[AblationResult] = []
    environment: EvaluationEnvironment | None = None
    for experiment in experiments:
        if environment is None or experiment.needs_fresh_index:
            if environment is not None and close is not None:
                close(environment)
            environment = environment_for(experiment)
        results.append(measure(environment, dataset, cases, experiment, k))
    if environment is not None and close is not None:
        close(environment)
    return tuple(results)
