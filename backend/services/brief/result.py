"""
The structured Research Brief: what a cross-Document result promises.

A brief that returns prose leaves a researcher with one job: trusting the
summary. The structured result replaces that trust with parts that can each
be checked: a summary to read, ordered claims to audit, the Passages that
support each claim kept apart from the Passages that disagree with it, the
questions the Documents could not settle, and whether the brief abstained
rather than completing from model knowledge.

The model writes the structure as a JSON block inside its final answer, the
same shape the chat path asks for with its claims block. The block is parsed
here, not in the loop, so the loop never sees a provider vocabulary. Every
id a claim names is checked against the evidence this run actually collected:
an id that names nothing held is dropped and the claim is marked unresolved
rather than shown as supported, because a false citation is worse than no
citation. Supporting and conflicting ids are never merged: a claim both
papers speak to stays contested on screen rather than averaged into
agreement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Sequence

#: The markers the model wraps its structured result in. The prose summary
#: stays outside the block so a reader who never parses JSON still gets the
#: answer, and the block itself is never shown.
BRIEF_OPEN = "<brief>"
BRIEF_CLOSE = "</brief>"

ClaimStatus = Literal["supported", "contested", "unresolved"]


@dataclass(frozen=True)
class BriefClaim:
    """One checkable statement, and the evidence on each side of it."""

    order: int
    text: str
    supports: tuple[str, ...]
    conflicts: tuple[str, ...]
    status: ClaimStatus

    def to_dict(self) -> dict[str, Any]:
        """Return the claim as the terminal event and the client read it."""
        return {
            "order": self.order,
            "claim": self.text,
            "supports": list(self.supports),
            "conflicts": list(self.conflicts),
            "status": self.status,
        }


@dataclass(frozen=True)
class StructuredBrief:
    """A parsed brief before its citations are checked."""

    summary: str
    claims: tuple[BriefClaim, ...]
    gaps: tuple[str, ...]
    abstained: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the brief as the terminal event records it."""
        return {
            "summary": self.summary,
            "claims": [claim.to_dict() for claim in self.claims],
            "gaps": list(self.gaps),
            "abstained": self.abstained,
        }


def parse_structured_brief(text: str) -> StructuredBrief | None:
    """
    Return the structured brief inside one final answer, if it wrote one.

    A final answer without a block is not an error: it is how the loop wrote
    briefs before the structure existed, and the caller wraps it as a summary
    with no claims. A block that is not JSON is also not an error for the same
    reason — the summary outside it is still what the reader reads.
    """
    raw = text or ""
    start = raw.find(BRIEF_OPEN)
    if start == -1:
        candidate = raw.strip()
        if candidate.startswith("{"):
            parsed = _read_json(candidate)
            if parsed is not None:
                return parsed
        return None
    end = raw.find(BRIEF_CLOSE, start)
    body = raw[start + len(BRIEF_OPEN) : end if end != -1 else len(raw)]
    parsed = _read_json(body.strip())
    if parsed is not None:
        return parsed
    # A marked block that is not JSON still carries the summary outside it.
    summary = (raw[:start].strip() or raw.strip())[:2000]
    return StructuredBrief(summary=summary, claims=(), gaps=(), abstained=False)


def _read_json(body: str) -> StructuredBrief | None:
    """Return one JSON body as a brief, or None when it is not one."""
    try:
        declared = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(declared, dict):
        return None
    summary = declared.get("summary")
    if not isinstance(summary, str):
        # A brief without a summary is prose the reader cannot skim, so it
        # is not a brief: the caller falls back to the answer text instead.
        return None
    claims = _read_claims(declared.get("claims"))
    gaps = _read_gaps(declared.get("gaps"))
    abstained = declared.get("abstained") is True
    return StructuredBrief(
        summary=summary.strip()[:2000],
        claims=claims,
        gaps=gaps,
        abstained=abstained,
    )


def _read_claims(declared: Any) -> tuple[BriefClaim, ...]:
    """Return the declared claims in order, dropping lines that are not claims."""
    if not isinstance(declared, list):
        return ()
    claims: list[BriefClaim] = []
    for position, entry in enumerate(declared, start=1):
        claim = _read_claim(entry, position)
        if claim is not None:
            claims.append(claim)
    return tuple(claims)


def _read_claim(entry: Any, order: int) -> BriefClaim | None:
    """Return one declared claim, or None when the line is not one."""
    if not isinstance(entry, dict):
        return None
    text = entry.get("claim")
    if not isinstance(text, str) or not text.strip():
        return None
    supports = _normalize_ids(entry.get("supports", entry.get("support", [])))
    conflicts = _normalize_ids(entry.get("conflicts", entry.get("conflict", [])))
    raw_status = entry.get("status")
    status: ClaimStatus = (
        raw_status
        if raw_status in ("supported", "contested", "unresolved")
        else _derive_status(supports, conflicts)
    )
    return BriefClaim(
        order=order,
        text=text.strip()[:2000],
        supports=supports,
        conflicts=conflicts,
        status=status,
    )


def _read_gaps(declared: Any) -> tuple[str, ...]:
    """Return the declared gaps, dropping entries that are not text."""
    if not isinstance(declared, list):
        return ()
    gaps: list[str] = []
    for entry in declared:
        if isinstance(entry, str) and entry.strip():
            gaps.append(entry.strip()[:1000])
        elif isinstance(entry, dict) and isinstance(entry.get("gap"), str):
            text = str(entry["gap"]).strip()
            if text:
                gaps.append(text[:1000])
    return tuple(gaps)


def _normalize_ids(declared: Any) -> tuple[str, ...]:
    """Return declared ids as the ledger spells them, in order, without repeats."""
    if not isinstance(declared, list):
        return ()
    seen: list[str] = []
    for item in declared:
        if not isinstance(item, str):
            continue
        evidence_id = item.strip().upper()
        if evidence_id and evidence_id not in seen:
            seen.append(evidence_id)
    return tuple(seen)


def _derive_status(
    supports: tuple[str, ...], conflicts: tuple[str, ...]
) -> ClaimStatus:
    """
    Return what the evidence on each side says about a claim.

    A claim nothing speaks to is unresolved. A claim with evidence on both
    sides stays contested rather than averaged: disagreement between papers
    is the finding, not noise around one. Anything else is supported.
    """
    if not supports and not conflicts:
        return "unresolved"
    if supports and conflicts:
        return "contested"
    if conflicts and not supports:
        return "contested"
    return "supported"


def validate_structured_brief(
    brief: StructuredBrief, allowed_ids: Sequence[str]
) -> tuple[StructuredBrief, tuple[str, ...]]:
    """
    Check one brief's citations against the evidence this run collected.

    A claim keeps the ids that name held evidence and loses the ones that do
    not, on each side separately, so one invented id does not throw away a
    claim the rest of its citations support and supporting evidence is never
    merged into conflicting evidence. A claim left with nothing on either side
    is marked unresolved rather than shown as supported from model knowledge.
    """
    allowed = {evidence_id.strip().upper() for evidence_id in allowed_ids}
    invalid: list[str] = []
    kept: list[BriefClaim] = []
    for claim in brief.claims:
        supports = tuple(item for item in claim.supports if item in allowed)
        conflicts = tuple(item for item in claim.conflicts if item in allowed)
        invalid.extend(item for item in (*claim.supports, *claim.conflicts) if item not in allowed)
        # An explicit unresolved stays unresolved: the model abstaining on a
        # claim it could have cited is a signal, not a status to upgrade.
        # Anything else is derived from the valid ids, so a claim left with
        # nothing on either side reads unresolved rather than supported.
        if claim.status == "unresolved":
            status: ClaimStatus = "unresolved"
        elif not supports and not conflicts:
            status = "unresolved"
        else:
            status = _derive_status(supports, conflicts)
        kept.append(
            BriefClaim(
                order=claim.order,
                text=claim.text,
                supports=supports,
                conflicts=conflicts,
                status=status,
            )
        )
    validated = StructuredBrief(
        summary=brief.summary,
        claims=tuple(kept),
        gaps=brief.gaps,
        abstained=brief.abstained or not kept and not brief.summary.strip(),
    )
    # Dedupe invalid ids while keeping the order the model wrote them in.
    seen: list[str] = []
    for evidence_id in invalid:
        if evidence_id not in seen:
            seen.append(evidence_id)
    return validated, tuple(seen)


def brief_from_answer(answer: str, allowed_ids: Sequence[str]) -> tuple[StructuredBrief, tuple[str, ...]]:
    """
    Return the structured brief for one final answer, validated.

    An answer without a block becomes a summary with no claims rather than an
    error, which is what keeps briefs written before the structure readable.
    A claim with no valid evidence is marked unresolved rather than completed
    from model knowledge.
    """
    parsed = parse_structured_brief(answer or "")
    if parsed is None:
        summary = (answer or "").strip()[:2000]
        return (
            StructuredBrief(summary=summary, claims=(), gaps=(), abstained=False),
            (),
        )
    return validate_structured_brief(parsed, allowed_ids)
