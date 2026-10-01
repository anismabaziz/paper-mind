"""
Answering a question the evidence does not reach.

A question with nothing to ground it in has one honest answer, and a language
model is not needed to produce it. This module decides whether retrieval left
anything usable, names the reason when it did not, and supplies the text the
user reads. The chat path asks it right after retrieval and before any
generation request, so a question this paper cannot answer costs nothing and
cannot be mistaken for a model failure.

Two absences are told apart, because they are not the same thing to a user:
retrieval found nothing at all, and retrieval found something that cannot be
quoted or pointed at. Both end the exchange the same way, and neither ends in a
citation, because there is no Passage to cite.
"""

from dataclasses import dataclass
from typing import Any, Literal, Sequence

AbstentionReason = Literal["no_evidence", "evidence_unusable"]

# Retrieval returned no Passage for the question at all.
NO_EVIDENCE: AbstentionReason = "no_evidence"
# Retrieval returned Passages, but none of them can be read or cited.
EVIDENCE_UNUSABLE: AbstentionReason = "evidence_unusable"

# The exact words the user reads, kept in one place so history replays the same
# sentence the stream did.
ABSTENTION_MESSAGES: dict[AbstentionReason, str] = {
    NO_EVIDENCE: (
        "This paper has no passage that speaks to this question, so there is "
        "nothing to answer from. Try asking about something the paper covers."
    ),
    EVIDENCE_UNUSABLE: (
        "The passages this question matched could not be read or cited, so "
        "there is nothing to answer from. Try asking about something else, or "
        "reindex the paper if you have not reindexed it in a while."
    ),
}


@dataclass(frozen=True)
class Abstention:
    """One refusal to answer, with the reason and the counts behind it."""

    reason: AbstentionReason
    message: str
    retrieved: int
    usable: int

    def to_dict(self) -> dict[str, Any]:
        """Return the abstention as the event payload the client reads."""
        return {
            "abstained": True,
            "message": self.message,
            "reason": self.reason,
            "retrieved": self.retrieved,
        }


def is_usable(source: dict[str, Any]) -> bool:
    """
    Return whether one retrieved Passage can carry an answer.

    A Passage has to say something and to say where it came from: the second
    half is what lets the reader check the answer against the page. A result
    missing either cannot be quoted as evidence, whatever its score.
    """
    content = source.get("content")
    document = source.get("document")
    return bool(
        isinstance(content, str)
        and content.strip()
        and isinstance(document, str)
        and document.strip()
    )


def abstention_for(sources: Sequence[dict[str, Any]]) -> Abstention | None:
    """
    Return the abstention a request must record, or None when it has evidence.

    The counts travel with it so a later reader can tell an empty index from
    one that matched Passages it could not use.
    """
    usable = sum(1 for source in sources if is_usable(source))
    if usable:
        return None
    retrieved = len(sources)
    reason = EVIDENCE_UNUSABLE if retrieved else NO_EVIDENCE
    return Abstention(
        reason=reason,
        message=ABSTENTION_MESSAGES[reason],
        retrieved=retrieved,
        usable=usable,
    )
