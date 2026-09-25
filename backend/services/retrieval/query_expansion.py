"""
Deterministic query expansion for contextual follow-up questions.

An opt-in model-based rewriter can be plugged into the same call.

A follow-up such as "what about the second method?" carries almost no terms
that appear in the document, so retrieving on it alone finds nothing useful.
The deterministic path prepends the user's recent questions to the current one,
which resolves pronouns ("it", "that") and ordinal references ("the second
method") without paying for a model call.

The model-based rewriter is off unless a named setting turns it on, and it is
best-effort: any failure falls back to the deterministic expansion so a
rewriter outage degrades retrieval quality rather than the request.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

_WORD_RE = re.compile(r"[a-z0-9%°]+")

#: Words that point back at earlier turns instead of naming document content.
_PRONOUNS = frozenset(
    {
        "it",
        "its",
        "they",
        "them",
        "their",
        "this",
        "that",
        "these",
        "those",
        "he",
        "she",
        "him",
        "her",
    }
)

#: Ordinals and comparatives that refer back to a list the document defines.
_ORDINALS = frozenset(
    {
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "last",
        "final",
        "next",
        "other",
        "others",
        "another",
        "previous",
        "above",
        "earlier",
    }
)

#: Phrases that point at the exchange rather than at document content. Kept
#: narrow on purpose: a phrase like "again" fires on ordinary questions ("run
#: the training again") and would drag unrelated prior questions into the
#: embedding for no gain.
_REFERENCE_PHRASES = (
    "you said",
    "you mentioned",
    "as mentioned",
    "mentioned above",
)


@dataclass(frozen=True)
class QueryExpansion:
    """The original question, the query retrieval actually ran, and how."""

    original_query: str
    expanded_query: str
    method: str

    def to_dict(self) -> dict[str, str]:
        """Return the expansion as trace data."""
        return {
            "original_query": self.original_query,
            "expanded_query": self.expanded_query,
            "query_expansion": self.method,
        }


def _words(text: str) -> list[str]:
    """Split text into lowercase words, keeping stopwords."""
    return _WORD_RE.findall(text.lower())


def refers_to_prior_turns(query: str) -> bool:
    """Report whether a question refers back to earlier turns."""
    lowered = f" {query.lower()} "
    if any(phrase in lowered for phrase in _REFERENCE_PHRASES):
        return True
    return any(word in _PRONOUNS or word in _ORDINALS for word in _words(query))


def _join_questions(prior_questions: Sequence[str]) -> str:
    """Join recent user questions, oldest first, into one context string."""
    return " ".join(
        question.strip()
        for question in prior_questions
        if question and question.strip()
    )


def expand_query(
    query: str,
    prior_questions: Sequence[str],
    *,
    max_chars: int,
    rewrite: Callable[[str, str], str] | None = None,
) -> QueryExpansion:
    """
    Build the query retrieval runs for a follow-up question.

    ``prior_questions`` holds recent user questions, oldest first. With none, or
    a question that names its subject outright, the query is returned
    unchanged. When ``rewrite`` is supplied and succeeds, its output replaces
    the deterministic expansion; when it fails or returns nothing usable, the
    deterministic expansion stands.

    Truncation never drops the current question: the prior context is what
    gets shortened.
    """
    original = query
    prior = _join_questions(prior_questions)
    if not prior or not refers_to_prior_turns(query):
        return QueryExpansion(
            original_query=original, expanded_query=original, method="none"
        )

    if rewrite is not None:
        rewritten = _rewrite_or_none(rewrite, query, prior)
        if rewritten:
            return QueryExpansion(
                original_query=original,
                expanded_query=rewritten[:max_chars],
                method="model",
            )

    if max_chars <= 0:
        return QueryExpansion(
            original_query=original, expanded_query=original, method="deterministic"
        )
    room = max_chars - len(query) - 1
    if room < 0:
        # The question alone overruns the budget; it still goes to retrieval
        # whole, because a cut-off question retrieves worse, not better.
        return QueryExpansion(
            original_query=original, expanded_query=query, method="deterministic"
        )
    return QueryExpansion(
        original_query=original,
        expanded_query=f"{prior[-room:].lstrip()} {query}".strip(),
        method="deterministic",
    )


def _rewrite_or_none(
    rewrite: Callable[[str, str], str], query: str, prior: str
) -> str | None:
    """Run the rewriter, treating any failure or empty reply as no rewrite."""
    try:
        result = rewrite(query, prior)
    except Exception:
        return None
    if not isinstance(result, str) or not result.strip():
        return None
    return result.strip()
