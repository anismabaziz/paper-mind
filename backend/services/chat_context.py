"""
Bounded chat context for one chat request.

A Conversation grows without bound, so the request works from a window of the
most recent completed Turns. The window is bounded twice — by Turn count
and by a token budget — and the budget is applied from the oldest Turn forward,
so the newest exchange is the one that survives. Retrieved evidence gets its
own budget and is trimmed by dropping the lowest-ranked Citation Source first.

The current question is never truncated: it is the instruction the model
follows, and a shortened question answers something the user did not ask.

What the budget left out is reported through ``dropped_turns`` and
``dropped_sources`` rather than dropped silently.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import tiktoken

from services.llm.base import LLMProvider
from services.retrieval.query_expansion import QueryExpansion

# cl100k_base is the tokenizer the chunker already uses; stable, no download.
_ENCODING = tiktoken.get_encoding("cl100k_base")

REWRITE_INSTRUCTION = (
    "Rewrite the follow-up question as one self-contained search query. "
    "Resolve pronouns and ordinal references using the earlier questions. "
    "Reply with the query text only: no preamble, no quotation marks, no "
    "explanation. If the question is already self-contained, reply with it "
    "unchanged."
)

# Room held back for the system instruction and the prompt's section tags, so
# a model's read budget is never spent entirely on evidence.
_PROMPT_OVERHEAD = 256


def build_model_rewriter(provider: LLMProvider) -> Callable[[str, str], str]:
    """
    Return a rewriter backed by a chat provider's one-shot generation.

    It costs an extra model call per question, so the route only builds one
    when the named setting enables rewriting. A provider that cannot answer
    returns a fixed abstention string rather than raising; that string is not
    a rewritten query, so it is reported as no rewrite and the deterministic
    expansion stands.
    """

    def rewrite(query: str, prior: str) -> str:
        # The provider's generate surface is question-plus-context, so the
        # rewriter instruction carries both roles and the evidence slot is
        # empty: a rewrite must not borrow the answer context.
        rewritten = provider.generate_response(
            f"{REWRITE_INSTRUCTION}\n\n"
            f"Earlier questions:\n{prior}\n\n"
            f"Follow-up question: {query}",
            "",
        )
        if rewritten.strip() == LLMProvider.FALLBACK_ANSWER:
            return ""
        return rewritten

    return rewrite


def token_count(text: str) -> int:
    """Return the token count of a piece of text."""
    return len(_ENCODING.encode(text)) if text else 0


def render_prior_turns(turns: Sequence[dict[str, Any]]) -> str:
    """Render completed Turns as a plain exchange transcript."""
    lines: list[str] = []
    for turn in turns:
        question = (turn.get("question") or "").strip()
        answer = (turn.get("answer") or "").strip()
        if question:
            lines.append(f"User: {question}")
        if answer:
            lines.append(f"Assistant: {answer}")
    return "\n".join(lines)


@dataclass(frozen=True)
class ChatContext:
    """The prompt material for one chat request, plus what it left out."""

    query: str
    context: str
    prior_turns: str
    sources: tuple[dict[str, Any], ...]
    expansion: QueryExpansion
    dropped_turns: int
    dropped_sources: int


def bounded_turns(
    turns: Sequence[dict[str, Any]],
    max_turns: int,
    token_budget: int,
    turns_in_conversation: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """
    Return the newest Turns that fit the window, oldest first.

    The count limit is applied first, then the token budget walks the window
    from its oldest end, dropping Turns until the remainder fits. Returns the
    kept Turns and how many of the Conversation's Turns the model did not see.
    ``turns_in_conversation`` is the Conversation's full count of eligible
    Turns, so a window narrowed before this function still reports honestly;
    without it, only the drops visible here are counted.
    """
    window = list(turns)[-max_turns:] if max_turns > 0 else []
    dropped = len(turns) - len(window)
    while window and token_count(render_prior_turns(window)) > token_budget:
        window.pop(0)
        dropped += 1
    unseen = (turns_in_conversation - len(window)) if turns_in_conversation else dropped
    return window, max(unseen, dropped)


def bounded_sources(
    sources: Sequence[dict[str, Any]], token_budget: int
) -> tuple[list[dict[str, Any]], int]:
    """
    Return the highest-ranked Citation Sources that fit the context budget.

    Sources arrive ranked, so a budget overrun drops the tail rather than
    splitting a Passage. Returns the kept sources and how many were dropped.
    """
    kept: list[dict[str, Any]] = []
    used = 0
    for source in sources:
        cost = token_count(source.get("content") or "")
        if kept and used + cost > token_budget:
            break
        kept.append(source)
        used += cost
    return kept, len(sources) - len(kept)


def build_chat_context(
    query: str,
    sources: Sequence[dict[str, Any]],
    turns: Sequence[dict[str, Any]],
    expansion: QueryExpansion,
    *,
    max_turns: int,
    prior_turns_token_budget: int,
    context_token_budget: int,
    turns_in_conversation: int | None = None,
    input_token_budget: int | None = None,
) -> ChatContext:
    """
    Assemble the bounded evidence, transcript, and question for one request.

    ``input_token_budget`` is the chosen model's own read budget. The app's
    evidence and transcript budgets are what this codebase thinks is useful;
    the model's window is what the provider will actually accept. When the
    model's budget is the tighter of the two, the evidence budget shrinks to
    whatever the question and transcript leave behind — and because evidence
    is dropped from the lowest-ranked end, the best passages are the ones that
    survive.
    """
    kept_turns, dropped_turns = bounded_turns(
        turns, max_turns, prior_turns_token_budget, turns_in_conversation
    )
    prior_turns_text = render_prior_turns(kept_turns)
    evidence_budget = context_token_budget
    if input_token_budget is not None:
        # The system instruction and the prompt's section tags ride along with
        # the question, so what is left is an over-estimate, never an
        # under-estimate.
        spent = token_count(query) + token_count(prior_turns_text) + _PROMPT_OVERHEAD
        evidence_budget = max(0, min(context_token_budget, input_token_budget - spent))
    kept_sources, dropped_sources = bounded_sources(sources, evidence_budget)
    return ChatContext(
        query=query,
        context="\n\n".join(source.get("content") or "" for source in kept_sources),
        prior_turns=prior_turns_text,
        sources=tuple(kept_sources),
        expansion=expansion,
        dropped_turns=dropped_turns,
        dropped_sources=dropped_sources,
    )
