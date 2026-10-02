"""The stream contract both run paths serve: refusals, events, encoding."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Refusal:
    """
    Why a run never started, before any stream started.

    A value rather than an exception, because the evaluator reads the same refusal
    as a case outcome: a Document that cannot be asked about is never scored as
    though it had produced a poor answer. A route that gets one hands it to
    ``routes.common.raise_refusal``, which turns it into the failure the client
    reads, so nothing here has to know what that shape is.
    """

    status: int
    category: str
    error: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnswerEvent:
    """One thing that happened to a question, named the way the stream names it."""

    name: str
    payload: dict[str, Any]

    def as_server_sent_event(self) -> str:
        """
        Return this event as the one server-sent event block it is.

        Both streams the application serves — a chat answer and a Research
        Brief — write the same framing, and the client parses the same framing.
        Encoding it here rather than in each route is what keeps the two from
        drifting into protocols a client can read one of and not the other.
        """
        return f"event: {self.name}\ndata: {json.dumps(self.payload)}\n\n"


def record_refusal(trace: Any, refusal: Refusal) -> Refusal:
    """
    Record a refused run and return the refusal unchanged.

    One helper rather than one per flow, so the trace always carries the same
    refusal the route and the evaluator read.
    """
    trace.refused(refusal.category, refusal.status)
    return refusal
