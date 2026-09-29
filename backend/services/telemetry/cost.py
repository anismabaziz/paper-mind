"""
What one answer call costs, in tokens and in dollars.

A trace that reports a latency but no cost is half a record: the number that
decides whether a change to the prompt or the retrieval is worth making is what
the call billed. The prices come from the model catalog the rest of the app
reads, so a trace cannot quote a price the application does not serve, and the
arithmetic lives here so the evaluation report and the trace cannot disagree
about what one case cost.
"""

from __future__ import annotations

from typing import Protocol


class PricedModel(Protocol):
    """What a cost estimate reads: the model's identity and its two prices."""

    id: str
    input_cost_per_million_usd: float
    output_cost_per_million_usd: float


def cost_usd(
    input_tokens: int | None,
    output_tokens: int | None,
    model: PricedModel,
) -> float | None:
    """
    Return what one call cost, in USD, at the catalog's published prices.

    The token counts are the app's own estimates, so this is an estimate too:
    exact arithmetic over estimated usage, not a provider invoice. A call with
    no measured usage has no cost rather than a cost of zero.
    """
    if input_tokens is None or output_tokens is None:
        return None
    return (
        input_tokens / 1_000_000 * model.input_cost_per_million_usd
        + output_tokens / 1_000_000 * model.output_cost_per_million_usd
    )
