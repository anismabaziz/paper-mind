"""LLM-as-judge faithfulness scoring.

The judge protocol is deliberately simple so any provider (or a fake) can
fill the judge role: given a judge prompt, the model must answer with a
single word — faithful, partial, or unfaithful. Parsing that word into a
score lives here; calling the model lives behind the injected callable.
"""

import re

FAITHFUL_PROMPT = """You are grading the faithfulness of an answer.

Question: {question}

Context the answer was generated from:
{context}

Answer to grade: {answer}

Is every claim in the answer supported by the context? Reply with exactly
one word: "faithful", "partial" (some claims unsupported or hedged beyond
the context), or "unfaithful" (the answer states things the context does
not contain)."""

SCORES = {"faithful": 1.0, "partial": 0.5, "unfaithful": 0.0}


def parse_verdict(judge_output: str) -> tuple[str, float]:
    """Extract the verdict word and its score from a judge reply."""
    match = re.search(
        r"\b(faithful|partial|unfaithful)\b", judge_output.lower()
    )
    if match is None:
        return "unparseable", 0.0
    verdict = match.group(1)
    return verdict, SCORES[verdict]


def judge_faithfulness(
    question: str, answer: str, context: str, judge
) -> tuple[str, float]:
    """Grade one answer. ``judge`` maps the prompt text to a model reply."""
    return parse_verdict(
        judge(FAITHFUL_PROMPT.format(question=question, context=context, answer=answer))
    )
