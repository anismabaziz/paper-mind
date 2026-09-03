"""Retrieval metrics over a ground-truth fixture.

All functions are pure and deterministic: they take retrieved chunk texts
and gold snippets and return numbers. No LLM, no I/O.
"""

from dataclasses import dataclass


def _contains_any(text: str, snippets: list[str]) -> bool:
    normalized = " ".join(text.lower().split())
    return any(
        " ".join(snippet.lower().split()) in normalized for snippet in snippets
    )


def hit_at_k(retrieved: list[str], gold_snippets: list[str], k: int) -> bool:
    """True if any of the top-k retrieved chunks contains a gold snippet."""
    return _contains_any(" \n".join(retrieved[:k]), gold_snippets)


def recall_at_k(retrieved: list[str], gold_snippets: list[str], k: int) -> float:
    """Fraction of gold snippets found within the top-k retrieved chunks."""
    if not gold_snippets:
        return 0.0
    top_k = " \n".join(retrieved[:k])
    found = sum(1 for snippet in gold_snippets if _contains_any(top_k, [snippet]))
    return found / len(gold_snippets)


@dataclass
class RetrievalReport:
    questions: int
    k: int
    hit_rate: float
    recall: float


def summarize(
    question_results: list[dict], k: int
) -> RetrievalReport:
    """Aggregate per-question results into hit-rate/recall@k.

    Each result dict needs a ``hit_at_k`` bool and a ``recall_at_k`` float.
    """
    if not question_results:
        return RetrievalReport(questions=0, k=k, hit_rate=0.0, recall=0.0)
    return RetrievalReport(
        questions=len(question_results),
        k=k,
        hit_rate=sum(r["hit_at_k"] for r in question_results)
        / len(question_results),
        recall=sum(r["recall_at_k"] for r in question_results)
        / len(question_results),
    )
