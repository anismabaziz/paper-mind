"""
Comparing evidence a brief already holds, without collecting more.

A model that has read Passages from two papers needs to say where they agree,
where they differ, and where they contradict each other. That judgement belongs
in the brief it writes — but the material it judges has to be arranged first:
which evidence is on the table, what vocabulary they share, and which pairs
look like they pull in opposite directions. That arranging is what this module
does, deterministically, so the same evidence always compares the same way and
the comparison never smuggles in a Passage the tools never returned.

Every entry it returns links back to the evidence ids it came from, and it
admits nothing: the ledger it read is the ledger the brief still holds. The
similarities are shared topic terms between pairs, the differences are where
the evidence comes from and which terms only one side uses, and the
contradictions are candidate contrasts — pairs that share a topic while one
side carries a negation or contrast cue. A candidate is a prompt to judge, not
a verdict: the model decides in the brief whether the pair truly disagrees.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from typing import Any

#: Words too common to say two Passages are about the same thing. Contrast
#: cues are deliberately not among them: the word that flags a candidate
#: contrast stays visible in the term arithmetic rather than being stripped
#: before the pair that carries it is ever compared.
_STOPWORDS = frozenset(
    """
    a an the and or but of to in on for with as at by from that this these
    those is are was were be been being has have had hasnt havent isnt arent
    wasnt werent it its it’s its their theirs them they he she we you i our
    your his her our ours theirs which who whom what when where how why so
    than too very can will just should now into over under between
    through during each other more most such only own same also within without
    may might must shall could would about across per via using used use uses
    based both either therefore
    thus hence et al fig figure table page pages section sections
    """.split()
)

#: Words that suggest one side pulls against the other rather than alongside
#: it. They mark a candidate contrast, never a proven contradiction.
_CONTRAST_CUES = frozenset(
    """
    not no never neither nor however although though whereas while unlike
    contradicts contradict contradicts contrary inconsistent inconsistency
    inconsistent differs differ different disagreement disagree disagrees
    fails failed failure lacks lacking lack absence absent limited limitation
    limitations insufficient except despite conversely instead rather
    """.split()
)

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

#: How many shared topic terms make a pair worth naming as similar.
_SHARED_TERMS_FOR_SIMILARITY = 5
#: How many shared topic terms make a cue-carrying pair worth flagging.
_SHARED_TERMS_FOR_CONTRAST = 3
#: How many terms each list carries, so a comparison stays bounded.
_MAX_TERMS = 8
#: How much of each evidence the comparison quotes, so one call cannot cost
#: the brief a whole Document.
QUOTED_CHARS = 2000


def significant_terms(text: str) -> tuple[str, ...]:
    """
    Return the topic terms of one text, in first-seen order, without repeats.

    Lowercased alphanumeric words of length three or more, without the
    stopwords that every academic Passage shares. What is left is what the
    Passage is about rather than how it is written.
    """
    seen: list[str] = []
    for match in _WORD.findall(text.lower()):
        if len(match) < 3 or match in _STOPWORDS or match in seen:
            continue
        seen.append(match)
    return tuple(seen)


def compare_items(items: Sequence[Any]) -> dict[str, Any]:
    """
    Arrange held evidence side by side, linked to the ids it came from.

    ``items`` are :class:`~services.brief.evidence.Evidence` the brief holds.
    Returns quoted text plus three lists — similarities, differences, and
    candidate contradictions — where every entry names the evidence ids it was
    drawn from and no entry names an id it was not given.
    """
    held = list(items)
    quoted = [
        {
            "evidence_id": item.evidence_id,
            "label": item.label,
            "page": item.page,
            "text": item.content[:QUOTED_CHARS],
            "truncated": len(item.content) > QUOTED_CHARS,
        }
        for item in held
    ]
    terms = {item.evidence_id: significant_terms(item.content) for item in held}
    return {
        "compared": quoted,
        "similarities": _similarities(held, terms),
        "differences": _differences(held, terms),
        "contradictions": _contradictions(held, terms),
        "note": (
            "A surface comparison of the named evidence only: similarities "
            "share topic terms, differences name provenance and one-sided "
            "terms, and contradictions are candidate contrasts to judge in "
            "the brief, not verdicts. No new evidence was collected."
        ),
    }


def _pairs(
    held: Sequence[Any], terms: dict[str, tuple[str, ...]]
) -> Iterator[tuple[Any, Any, list[str]]]:
    """
    Yield each unordered pair of held evidence with the terms they share.

    One place where pairs meet, so similarities and candidate contradictions
    always mean the same thing by sharing a topic: the order each side first
    used the terms in.
    """
    for left_pos, left in enumerate(held):
        for right in held[left_pos + 1 :]:
            shared = [
                term
                for term in terms[left.evidence_id]
                if term in set(terms[right.evidence_id])
            ]
            yield left, right, shared


def _similarities(
    held: Sequence[Any], terms: dict[str, tuple[str, ...]]
) -> list[dict[str, Any]]:
    """
    Return the pairs that share enough topic terms to be about the same thing.

    Each entry names the pair and up to a few of the terms they share, so the
    model can see what the overlap is rather than taking a bare claim of it.
    """
    found: list[dict[str, Any]] = []
    for left, right, shared in _pairs(held, terms):
        if len(shared) < _SHARED_TERMS_FOR_SIMILARITY:
            continue
        found.append(
            {
                "evidence_ids": [left.evidence_id, right.evidence_id],
                "shared_terms": shared[:_MAX_TERMS],
                "detail": (
                    f"{left.evidence_id} and {right.evidence_id} share "
                    f"{len(shared)} topic terms "
                    f"({', '.join(shared[:_MAX_TERMS])})."
                ),
            }
        )
    return found


def _differences(
    held: Sequence[Any], terms: dict[str, tuple[str, ...]]
) -> list[dict[str, Any]]:
    """
    Return where the evidence comes from and what only one side says.

    The first entry is provenance: which labelled Documents and Pages are on
    the table, because evidence from A page 2 and evidence from B page 9
    differ before a single word is compared. After it, each evidence gets the
    terms none of the others use.
    """
    if not held:
        return []
    labels = sorted({item.label for item in held})
    pages = sorted(
        {(item.label, item.page) for item in held},
        key=lambda pair: (pair[0], pair[1] if pair[1] is not None else -1),
    )
    page_words = ", ".join(
        f"[{label}] p. {page}" if page is not None else f"[{label}] unpaged"
        for label, page in pages
    )
    entries: list[dict[str, Any]] = [
        {
            "evidence_ids": [item.evidence_id for item in held],
            "detail": (
                f"Evidence spans {len(labels)} labelled Document(s) "
                f"({', '.join(labels)}): {page_words}."
            ),
        }
    ]
    for item in held:
        others = set()
        for other in held:
            if other.evidence_id != item.evidence_id:
                others |= set(terms[other.evidence_id])
        distinctive = [
            term for term in terms[item.evidence_id] if term not in others
        ][:_MAX_TERMS]
        if distinctive:
            entries.append(
                {
                    "evidence_ids": [item.evidence_id],
                    "detail": (
                        f"Terms only {item.evidence_id} uses: "
                        f"{', '.join(distinctive)}."
                    ),
                }
            )
    return entries


def _contradictions(
    held: Sequence[Any], terms: dict[str, tuple[str, ...]]
) -> list[dict[str, Any]]:
    """
    Return the pairs that share a topic while one side pushes back.

    A pair qualifies when it shares enough terms to be about the same thing,
    comes from different labelled Documents, and one side carries a negation
    or contrast cue. Each entry names the cue found, so the model judges the
    pair with the reason it was flagged in view.
    """
    found: list[dict[str, Any]] = []
    for left, right, shared in _pairs(held, terms):
        if left.label == right.label:
            continue
        if len(shared) < _SHARED_TERMS_FOR_CONTRAST:
            continue
        cue = _contrast_cue(left.content) or _contrast_cue(right.content)
        if cue is None:
            continue
        first, second = (
            (left, right) if _contrast_cue(left.content) else (right, left)
        )
        found.append(
            {
                "evidence_ids": [left.evidence_id, right.evidence_id],
                "shared_terms": shared[:_MAX_TERMS],
                "detail": (
                    f"Candidate contrast: {first.evidence_id} carries "
                    f"'{cue}' while sharing topic terms with "
                    f"{second.evidence_id} "
                    f"({', '.join(shared[:_MAX_TERMS])}). Judge whether "
                    "the two truly disagree in the brief."
                ),
            }
        )
    return found


def _contrast_cue(text: str) -> str | None:
    """Return the first contrast cue one text carries, if it carries one."""
    for match in _WORD.findall(text.lower()):
        if match in _CONTRAST_CUES:
            return match
    return None
