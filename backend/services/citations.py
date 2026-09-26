"""
Claim-level citations: what in an answer each retrieved Passage supports.

A list of retrieved Passages under an answer says only that the model had
something to read. It does not say which sentence came from which Passage, so
the reader has to take the whole answer on trust. This module is the contract
that fixes that: every Passage carries a stable source ID before the model is
called, the model is asked to name the IDs behind each claim it makes, and every
ID it writes is checked against the evidence that was actually supplied.

An ID that names nothing supplied is a citation to a Passage that does not
exist. It is never stored and never shown. It is either repaired once or the
turn fails, because a false citation is worse than no citation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from services.abstention import ABSTENTION_MESSAGES
from services.llm.base import LLMProvider

#: Bumped whenever the claims instruction changes what the model is asked for.
#: Stored on nothing, returned on every answer, and read by the traces: an
#: answer is only comparable to another answer if both were asked the same way.
PROMPT_VERSION = "grounded-claims-v1"

CLAIMS_OPEN = "<claims>"
CLAIMS_CLOSE = "</claims>"

#: The citation contract, appended to the system instruction.
CLAIMS_INSTRUCTION = (
    "Cite every factual claim. After the answer, add a claims block: the line "
    f"{CLAIMS_OPEN}, then one JSON object per line with the keys \"claim\" and "
    "\"sources\", then the line "
    f"{CLAIMS_CLOSE}. Each claim is one statement you made, and its sources "
    "are the ids of the retrieved passages that support it, taken only from "
    "the bracketed ids in the context such as [S1]. Never invent an id, never "
    "cite a passage that does not appear in the context, and give a claim an "
    "empty sources list when no passage supports it. Write nothing outside the "
    "block except the answer itself."
)

#: Why a turn is closed when a citation survives one repair still naming
#: nothing the app supplied.
UNRESOLVED_CITATIONS_REASON = "invalid citations"


@dataclass(frozen=True)
class Claim:
    """One statement the answer makes, and the Passages said to support it."""

    claim: str
    source_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the claim as the client and the store read it."""
        return {"claim": self.claim, "sources": list(self.source_ids)}


@dataclass(frozen=True)
class ParsedAnswer:
    """A generated answer split into the prose to show and the claims to check."""

    answer: str
    claims: tuple[Claim, ...]


@dataclass(frozen=True)
class ValidatedCitations:
    """Claims that survived checking, and what checking rejected."""

    claims: tuple[Claim, ...]
    invalid_ids: tuple[str, ...]
    grounded: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the validated citations as the terminal event payload."""
        return {
            "claims": [claim.to_dict() for claim in self.claims],
            "grounded": self.grounded,
            "prompt_version": PROMPT_VERSION,
        }


def assign_source_ids(sources: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Return the retrieved Passages with a stable id and a rank each.

    The id is fixed before the model is called, because the model may only
    cite ids it was shown. The rank is the place the Passage took in
    retrieval, which is the honest thing to show a reader: a raw similarity
    score is a ranking signal, not a percentage of being right.
    """
    return [
        {**source, "source_id": f"S{position}", "rank": position}
        for position, source in enumerate(sources, start=1)
    ]


def render_evidence(sources: Iterable[dict[str, Any]]) -> str:
    """Return the evidence as the model reads it, with each Passage's id."""
    return "\n\n".join(
        f"[{source['source_id']}] {source.get('content') or ''}"
        for source in sources
    )


class AnswerSplitter:
    """
    Split a generated stream into the prose to show and the claims to check.

    The claims block is the app's own request, written in the model's own
    output, so the reader must never see it: a line of JSON under an answer
    looks like the app leaking its bookkeeping. A marker can also arrive split
    across two fragments, so the tail that could still become one is held back
    until the next fragment settles it.
    """

    def __init__(self) -> None:
        """Start with nothing held back and no claims block seen."""
        self._pending = ""
        self._in_claims = False

    def feed(self, fragment: str) -> str:
        """Return the part of one fragment that is safe to put on screen."""
        if self._in_claims:
            return ""
        self._pending += fragment
        start = self._pending.find(CLAIMS_OPEN)
        if start != -1:
            visible = self._pending[:start]
            self._pending = ""
            self._in_claims = True
            return visible
        # Hold back only what could still turn into the marker.
        keep = len(CLAIMS_OPEN) - 1
        if len(self._pending) <= keep:
            return ""
        visible = self._pending[:-keep]
        self._pending = self._pending[-keep:]
        return visible

    def finish(self) -> str:
        """Return whatever prose was still held back, and end the split."""
        visible = "" if self._in_claims else self._pending
        self._pending = ""
        self._in_claims = False
        return visible


def claims_block(claims: Sequence[dict[str, Any]]) -> str:
    """Return claims in the exact shape a model is asked to write them."""
    lines = "\n".join(json.dumps(claim) for claim in claims)
    return f"{CLAIMS_OPEN}\n{lines}\n{CLAIMS_CLOSE}"


def parse_claims(text: str) -> ParsedAnswer:
    """
    Split one generated answer into its prose and the claims it declared.

    A line in the block that is not a claim the app can read is not a claim:
    it is dropped rather than guessed at, so a model that writes prose inside
    the block cannot smuggle an uncitable claim through it.
    """
    raw = (text or "").strip()
    start = raw.find(CLAIMS_OPEN)
    if start == -1:
        return ParsedAnswer(answer=raw, claims=())
    answer = raw[:start].strip()
    end = raw.find(CLAIMS_CLOSE, start)
    body = raw[start + len(CLAIMS_OPEN) : end if end != -1 else len(raw)]
    claims: list[Claim] = []
    for line in body.splitlines():
        if not line.strip():
            continue
        claim = _read_claim(line)
        if claim is not None:
            claims.append(claim)
    return ParsedAnswer(answer=answer, claims=tuple(claims))


def _read_claim(line: str) -> Claim | None:
    """Return one declared claim, or None when the line is not one."""
    try:
        declared = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(declared, dict):
        return None
    text = declared.get("claim")
    if not isinstance(text, str) or not text.strip():
        return None
    declared_ids = declared.get("sources")
    if not isinstance(declared_ids, list):
        return None
    return Claim(claim=text.strip(), source_ids=_normalize_ids(declared_ids))


def _normalize_ids(declared_ids: list[Any]) -> tuple[str, ...]:
    """Return the declared ids as the app spells them, in order, without repeats."""
    seen: list[str] = []
    for declared in declared_ids:
        if not isinstance(declared, str):
            continue
        source_id = declared.strip().upper()
        if source_id and source_id not in seen:
            seen.append(source_id)
    return tuple(seen)


def validate_claims(
    claims: Sequence[Claim], allowed_ids: Iterable[str]
) -> ValidatedCitations:
    """
    Check declared citations against the evidence that was actually supplied.

    A claim keeps the ids that name a supplied Passage and loses the ones that
    do not, so one invented id does not throw away a claim the rest of its
    citations support. Every rejected id is reported: the caller decides
    whether to ask the model to repair them or to close the turn.
    """
    allowed = {source_id.strip().upper() for source_id in allowed_ids}
    kept: list[Claim] = []
    invalid: list[str] = []
    for claim in claims:
        supported = tuple(
            source_id for source_id in claim.source_ids if source_id in allowed
        )
        invalid.extend(
            source_id for source_id in claim.source_ids if source_id not in allowed
        )
        kept.append(Claim(claim=claim.claim, source_ids=supported))
    return ValidatedCitations(
        claims=tuple(kept),
        invalid_ids=tuple(invalid),
        grounded=grounded_in_claims(kept),
    )


def grounded_in_claims(claims: Sequence[Claim]) -> bool:
    """Report whether at least one claim names a supplied Passage."""
    return any(claim.source_ids for claim in claims)


def prune_conflicting_claims(answer: str, claims: Sequence[Claim]) -> tuple[Claim, ...]:
    """
    Return the claims that do not contradict the answer they arrived with.

    A model that says it cannot answer and then cites evidence for a claim it
    made has produced two contradictory statements. The claims go: the reader
    is shown the refusal, which is the one the app can stand behind.
    """
    if _abstains(answer) and claims:
        return ()
    return tuple(claims)


def _abstains(answer: str) -> bool:
    """Report whether an answer is the app's own wording for not knowing."""
    text = (answer or "").strip()
    if not text:
        return False
    known = (LLMProvider.FALLBACK_ANSWER, *ABSTENTION_MESSAGES.values())
    return any(text == message or text.startswith(message) for message in known)


def repair_instruction(
    answer: str, claims: Sequence[Claim], allowed_ids: Sequence[str]
) -> str:
    """
    Return the one question asked to fix citations that name nothing supplied.

    The answer itself is not rewritten — the reader has already read it, and a
    second version of the same sentences would be a different answer. Only the
    mapping from claim to Passage is asked for again, and it is asked for with
    the passages themselves in the context: reassigning a citation without
    reading the evidence is a guess, not a repair. The reply is read back by
    the same parser as the first one, so the block is requested in the same
    words the first block was.
    """
    listed = "\n".join(
        f'- "{claim.claim}" cited {list(claim.source_ids)}' for claim in claims
    )
    return (
        "The answer below cites source ids that were not supplied with this "
        "question, so those citations are false. The passages are given in the "
        "context under the ids you may cite.\n\n"
        "Return the same claims with corrected source ids, as a claims block: "
        f"the line {CLAIMS_OPEN}, then one JSON object per line with the keys "
        '"claim" and "sources", then the line '
        f"{CLAIMS_CLOSE}. No code fence, no preamble, no answer. Only these ids "
        f"exist: {', '.join(allowed_ids)}. A claim no passage supports must have "
        "an empty sources list, and the claim text stays exactly as it is.\n\n"
        f"Claims to correct:\n{listed}\n\n"
        f"Answer already shown to the reader:\n{answer}"
    )
