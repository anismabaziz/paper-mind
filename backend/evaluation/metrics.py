"""
The numbers a run reports, and nothing else.

Retrieval quality, latency, and cost are all computed here from values the run
already measured: the ranked chunks retrieval returned, the gold snippets the
case set declares, and the seconds a case took. Nothing in this module calls a
model, reads a clock, or touches a database, so any number a report publishes
can be recomputed from the case it came from.

Two conventions are fixed here and nowhere else. Percentiles are interpolated
between the two samples that straddle the requested rank, so p50 of an even
number of samples is the mean of the two middle ones — the definition every
latency dashboard uses, spelled out so two reports can be compared. And
cold-start measurements are summarized apart from steady-state ones, because a
run's first case pays for loading a model and nobody's second case does.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


class PricedModel(Protocol):
    """What a cost estimate reads: the model's identity and its two prices."""

    id: str
    input_cost_per_million_usd: float
    output_cost_per_million_usd: float


def _normalize(text: str) -> str:
    """Return text folded to one space and lower case, for comparison."""
    return " ".join(str(text or "").lower().split())


def contains_snippet(text: str, snippet: str) -> bool:
    """
    Report whether a Passage holds a snippet, ignoring case and line wrapping.

    Shared rather than written again per module, because a metric and a grader
    that disagreed about whether a chunk holds a snippet would produce a report
    whose two halves contradict each other.
    """
    return _normalize(snippet) in _normalize(text)


def _contains_any(text: str, snippets: Sequence[str]) -> bool:
    """Report whether a chunk contains at least one of the snippets."""
    return any(contains_snippet(text, snippet) for snippet in snippets)


def _relevant_chunks(
    retrieved: Sequence[str], gold: Sequence[str], k: int
) -> list[bool]:
    """Return, for each of the top-k chunks, whether it holds any gold snippet."""
    return [_contains_any(chunk, gold) for chunk in retrieved[:k]]


def hit_at_k(retrieved: Sequence[str], gold_snippets: Sequence[str], k: int) -> bool:
    """Return True if any of the top-k retrieved chunks contains a gold snippet."""
    return _contains_any(" \n".join(retrieved[:k]), gold_snippets)


def recall_at_k(
    retrieved: Sequence[str], gold_snippets: Sequence[str], k: int
) -> float:
    """Fraction of gold snippets found within the top-k retrieved chunks."""
    if not gold_snippets:
        return 0.0
    top_k = " \n".join(retrieved[:k])
    found = sum(1 for snippet in gold_snippets if _contains_any(top_k, [snippet]))
    return found / len(gold_snippets)


def reciprocal_rank(
    retrieved: Sequence[str], gold_snippets: Sequence[str], k: int
) -> float:
    """
    Return 1 divided by the rank of the first relevant chunk, or 0.0.

    Averaged over questions this is MRR, which rewards putting the passage a
    question needs at the top rather than merely inside the window.
    """
    for rank, chunk in enumerate(retrieved[:k], start=1):
        if _contains_any(chunk, gold_snippets):
            return 1.0 / rank
    return 0.0


def _dcg(relevance: Sequence[bool]) -> float:
    """Return the discounted gain of a ranking, one unit for a relevant chunk."""
    return sum(
        1.0 / math.log2(position + 1)
        for position, relevant in enumerate(relevance, start=1)
        if relevant
    )


def ndcg_at_k(retrieved: Sequence[str], gold_snippets: Sequence[str], k: int) -> float:
    """
    Return the normalized discounted gain of a ranking against the ideal one.

    A chunk is relevant or it is not, and it does not count twice for holding
    two gold snippets: whether one chunk could have held both depends on the
    chunk size, and a score that moved with the chunk size would not be
    comparable across runs that configured it differently.

    The ideal is one relevant chunk per gold snippet, so a run that retrieved
    only some of the evidence cannot reach 1.0 even with what it did retrieve at
    the top. That is the whole difference from hit rate: this measures how much
    of the evidence was found and how high up it sat, not whether the window
    contained anything at all.
    """
    relevance = _relevant_chunks(retrieved, gold_snippets, k)
    best = _dcg([True] * min(len(gold_snippets), k))
    if not any(relevance) or not best:
        return 0.0
    # A snippet that appears in two chunks makes more than the ideal number of
    # relevant chunks possible, and no ranking should score above perfect.
    return min(1.0, _dcg(relevance) / best)


def retrieval_scores(
    retrieved: Sequence[str], gold_snippets: Sequence[str], k: int
) -> dict[str, Any]:
    """Return one question's retrieval numbers, all four of them."""
    return {
        "hit_at_k": hit_at_k(retrieved, gold_snippets, k),
        "recall_at_k": recall_at_k(retrieved, gold_snippets, k),
        "reciprocal_rank": reciprocal_rank(retrieved, gold_snippets, k),
        "ndcg_at_k": ndcg_at_k(retrieved, gold_snippets, k),
    }


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean of a non-empty sequence."""
    return sum(values) / len(values)


@dataclass
class RetrievalReport:
    """Retrieval over the questions a run asked, per question and in aggregate."""

    questions: int
    k: int
    hit_rate: float
    recall: float
    mrr: float
    ndcg: float
    #: One row per question, in the order the run asked them, so a reader can
    #: see which question dragged an average down rather than only that it did.
    per_question: list[dict[str, Any]] = field(default_factory=list)


def summarize(
    question_results: Sequence[dict[str, Any]],
    k: int,
    questions: Sequence[str],
) -> RetrievalReport:
    """
    Aggregate per-question retrieval numbers into the run's report.

    ``questions`` names the results, in order, and is required: an aggregate
    whose rows cannot be traced back to the question that produced them is the
    number this whole module exists to make trustworthy. Fewer names than rows
    is a caller bug, not something to paper over.
    """
    if len(questions) != len(question_results):
        raise ValueError(
            f"{len(question_results)} retrieval results cannot be named by "
            f"{len(questions)} questions"
        )
    if not question_results:
        return RetrievalReport(
            questions=0,
            k=k,
            hit_rate=0.0,
            recall=0.0,
            mrr=0.0,
            ndcg=0.0,
            per_question=[],
        )
    rows = [{"id": name, **row} for name, row in zip(questions, question_results)]
    return RetrievalReport(
        questions=len(rows),
        k=k,
        hit_rate=_mean([float(row["hit_at_k"]) for row in rows]),
        recall=_mean([row["recall_at_k"] for row in rows]),
        mrr=_mean([row["reciprocal_rank"] for row in rows]),
        ndcg=_mean([row["ndcg_at_k"] for row in rows]),
        per_question=rows,
    )


def percentile(values: Sequence[float], quantile: float) -> float | None:
    """
    Return the requested quantile of a sample, interpolating between neighbours.

    The rank is ``quantile * (n - 1)``; the two samples that straddle it are
    weighted by how far between them the rank falls. An empty sample has no
    quantile, and says so with None rather than reporting 0.0 — which would read
    as the fastest run ever measured.
    """
    if not values:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile out of range: {quantile}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * weight)


@dataclass(frozen=True)
class LatencySummary:
    """How long something took, summarized as a median and a tail."""

    samples: int
    p50: float | None
    p95: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return the summary as a report stores it."""
        return asdict(self)


def latency_summary(samples: Sequence[float]) -> LatencySummary:
    """
    Summarize one population of samples as a median and a tail.

    A run keeps two of these: one over the cases asked after the first, and one
    over the first case itself. The caller holds them apart rather than passing
    them in here, because mixing them would report a slow first question as if
    every question were slow.
    """
    return LatencySummary(
        samples=len(samples),
        p50=percentile(samples, 0.5),
        p95=percentile(samples, 0.95),
    )


def cost_for(
    input_tokens: int | None,
    output_tokens: int | None,
    model: PricedModel,
) -> float | None:
    """
    Return what one case cost, in USD, at the catalog's published prices.

    The token counts are the app's own estimates, so this is an estimate too:
    it is exact arithmetic over estimated usage, not a provider invoice. A case
    with no measured usage has no cost rather than a cost of zero.
    """
    if input_tokens is None or output_tokens is None:
        return None
    return (
        input_tokens / 1_000_000 * model.input_cost_per_million_usd
        + output_tokens / 1_000_000 * model.output_cost_per_million_usd
    )


@dataclass
class MetricSummary:
    """
    One graded metric, in the shape every graded metric is reported in.

    ``graded`` counts the cases where the metric applied at all. ``scored`` is
    the smaller number of cases it produced a number for, because a grader may
    return Unknown: an answer nothing can be decided about is left out of the
    mean and counted on its own, never scored as a zero.
    """

    graded: int
    scored: int
    mean: float | None
    passed: int
    failed: int
    unknown: int

    def to_dict(self) -> dict[str, Any]:
        """Return the metric's counts and its mean."""
        return {
            "graded": self.graded,
            "scored": self.scored,
            "mean": self.mean,
            "passed": self.passed,
            "failed": self.failed,
            "unknown": self.unknown,
        }

    @classmethod
    def of(
        cls, scores: Sequence[float | None], outcomes: Sequence[str]
    ) -> "MetricSummary":
        """
        Return the summary of one metric over the cases that defined it.

        ``outcomes`` says pass, fail, or unknown for each score in the same
        order, so a metric that is a fraction rather than a grade — citation
        precision, say — reports its own counts. The report keys the summary by
        the metric's name, so the name is not repeated on the value.
        """
        scored = [score for score in scores if score is not None]
        return cls(
            graded=len(outcomes),
            scored=len(scored),
            mean=_mean(scored) if scored else None,
            passed=sum(1 for outcome in outcomes if outcome == "passed"),
            failed=sum(1 for outcome in outcomes if outcome == "failed"),
            unknown=sum(1 for outcome in outcomes if outcome == "unknown"),
        )
