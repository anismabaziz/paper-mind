"""
One Research Brief, recorded.

A brief is the first request here where the interesting record is not a single
call but a trajectory: what the model reached for, in what order, what came
back, what it chose to read, and what the loop was allowed to spend while it did.
An operator reading a failed brief needs all of that, and none of it is the
words of the question, the answer, or the Passages.

So the shape of a brief's trace is a span per tool call, plus one span for the
loop's own budget. Each tool span records the tool, whether it ran or was
refused, why it was refused, the arguments' shape but never their values, the
evidence ids it returned, and the retrieval ranks those Passages took. The budget
span records what was spent against every ceiling, so a brief that stopped on a
limit is legible from the record alone.

The redaction rules are the ones the answer path already follows, extended with
the attribute names a tool call introduces: ``arguments``, ``tool_arguments``,
and ``search_query`` hold model-written text that is derived from the reader's
question, and a trace is not where that belongs.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from services.brief.budget import BriefBudget
from services.brief.prompts import BRIEF_PROMPT_VERSION
from services.telemetry.spans import Span, Trace, Tracer, error_category

#: What a brief can end as, as a trace records it. The words are the ones the
#: terminal event uses, so the stream and the trace never disagree.
COMPLETE = "complete"
INCOMPLETE = "incomplete"
FAILED = "failed"
CANCELLED = "cancelled"
REFUSED = "refused"

#: The spans one brief contains, in the order they happen.
#: How much of a call's digest a trace keeps. Enough to tell two calls apart
#: in a list, short enough that it is not the call it stands for.
_FINGERPRINT_CHARS = 16


def _digest(value: str) -> str:
    """Return what may be written down about one call without writing it down."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:_FINGERPRINT_CHARS]


SCOPE = "scope"
TOOL = "tool"
BUDGET = "budget"
OUTCOME = "outcome"


class BriefTrace:
    """
    The trace of one brief run, and the vocabulary it is recorded in.

    A tracer with no exporter records nothing anywhere, and the calls that
    record a brief are the same ones either way, so a trace cannot describe a
    different run than the one the application served.
    """

    def __init__(self, tracer: Tracer | None, trace: Trace, model: Any = None) -> None:
        """Record this brief's steps into this trace."""
        self._tracer = tracer
        self._trace = trace
        self._model = model
        self._finished = False
        self._clock: Callable[[], float] = time.monotonic
        self._started_at: float | None = None
        #: Every tool span, so the trajectory is on the trace as well as on the
        #: spans: an operator reading only the trace attributes sees the order.
        self._trajectory: list[dict[str, Any]] = []

    @classmethod
    def start(
        cls, tracer: Tracer | None, *, model: Any = None, name: str = "research.brief"
    ) -> "BriefTrace":
        """Open a trace for one brief, naming the model that will run it."""
        trace = tracer.start(name) if tracer is not None else Trace(name=name)
        brief_trace = cls(tracer, trace, model)
        if model is not None:
            brief_trace.identify(
                provider=model.provider,
                model=model.id,
                prompt_version=BRIEF_PROMPT_VERSION,
            )
        return brief_trace

    @property
    def trace_id(self) -> str:
        """Return the identifier every span of this brief shares."""
        return self._trace.trace_id

    @property
    def trace(self) -> Trace:
        """Return the underlying trace, for callers that read it directly."""
        return self._trace

    def identify(self, **attributes: Any) -> None:
        """Record what this brief is about."""
        self._trace.identify(**attributes)

    def begin(self, clock: Callable[[], float]) -> None:
        """Begin the brief's own clock, before the first model turn."""
        self._clock = clock
        self._started_at = clock()

    def clock(self) -> float:
        """
        Return the brief's monotonic reading, or now when it has not begun.

        The loop and the trace share this so a span's latency is measured
        against the same clock the budget is, rather than against a second one
        read a moment apart.
        """
        return self._clock() if self._started_at is not None else time.monotonic()

    def elapsed(self) -> float:
        """Return how long this brief has run, in seconds."""
        if self._started_at is None:
            return 0.0
        return max(0.0, self._clock() - self._started_at)

    def refused(self, category: str, status: int) -> None:
        """Record a brief that was refused before any model turn ran."""
        self.identify(outcome=REFUSED, refusal_status=status)
        if category:
            self.identify(refusal_category=category)
        self.finish()

    def scope(self, documents: list[dict[str, Any]]) -> Span:
        """
        Open the span recording which Documents this brief was allowed to read.

        Labels, titles, and identifiers only. The pair is the entire reach of
        the run, so it belongs on the record even though nothing reads it back.
        """
        span = self._trace.span(SCOPE)
        span.record(document_count=len(documents), documents=documents)
        span.end()
        return span

    @contextmanager
    def tool(
        self,
        *,
        turn: int,
        name: str,
        arguments: dict[str, Any],
        fingerprint: str,
    ) -> Iterator[Span]:
        """
        Open the span one tool call is recorded into.

        The arguments are handed in so the call's shape can be recorded, and only
        the shape is: a search query is the model's paraphrase of the reader's
        question, and a trace that repeated it would be a second copy of the
        question sitting in a file. The fingerprint goes on the span as a digest
        for the same reason — it identifies the call without carrying it — which
        is why the digest is computed here rather than recorded as it arrived.
        """
        span = self._trace.span(TOOL)
        span.record(
            turn=turn,
            tool=name,
            argument_keys=sorted(arguments),
            call_fingerprint=_digest(fingerprint),
        )
        try:
            yield span
        except Exception as error:  # noqa: BLE001 - the failure is the record
            span.fail(error_category(error))
            raise
        finally:
            span.end()
            self._trajectory.append(
                {
                    "turn": turn,
                    "tool": name,
                    "outcome": span.error_category or "ran",
                    "evidence_ids": span.attributes.get("evidence_ids", []),
                    "ranks": span.attributes.get("ranks", []),
                    "reason": span.attributes.get("refusal_reason"),
                }
            )

    def searched(
        self,
        span: Span,
        *,
        label: str,
        document_id: str,
        method: str,
        outcome: str,
        candidate_count: int,
        latency_ms: float,
        evidence: list[Any],
    ) -> None:
        """
        Record what one search returned, as ranks rather than as text.

        The evidence ids and the rank each Passage took are the whole record of
        a search: enough to see what the model was shown and in what order,
        without the trace holding a copy of either Document.
        """
        span.record(
            label=label,
            document_id=document_id,
            retrieval_method=method,
            retrieval_outcome=outcome,
            candidate_count=candidate_count,
            latency_ms=latency_ms,
            evidence_ids=[item.evidence_id for item in evidence],
            ranks=[
                {"evidence_id": item.evidence_id, "rank": item.rank, "page": item.page}
                for item in evidence
            ],
        )

    def refused_tool(self, span: Span, reason: str) -> None:
        """Record a call the brief would not make, and why."""
        span.record(refused=True, refusal_reason=reason)
        span.fail(reason)

    def read(self, span: Span, *, evidence_ids: list[str]) -> None:
        """Record which held Passages the model asked to read."""
        span.record(evidence_ids=evidence_ids)

    def paged(
        self,
        span: Span,
        *,
        evidence_id: str,
        label: str,
        document_id: str,
        page: int,
        page_count: int,
        latency_ms: float,
        truncated: bool,
    ) -> None:
        """
        Record what one Page read returned, as provenance rather than as text.

        The evidence id the Page was admitted under, its label, and its Page
        number are the whole record: enough to see what the model was shown,
        without the trace holding a copy of either Document.
        """
        span.record(
            label=label,
            document_id=document_id,
            evidence_ids=[evidence_id],
            ranks=[{"evidence_id": evidence_id, "rank": 1, "page": page}],
            page=page,
            page_count=page_count,
            latency_ms=latency_ms,
            truncated=truncated,
        )

    def compared(
        self, span: Span, *, evidence_ids: list[str], latency_ms: float = 0.0
    ) -> None:
        """Record which held evidence the model asked to compare."""
        span.record(evidence_ids=evidence_ids, latency_ms=latency_ms)

    def budget(
        self,
        budget: BriefBudget,
        *,
        stopped: str = "",
        evidence: Sequence[Any] = (),
    ) -> None:
        """
        Record what the loop spent, against every ceiling it was given.

        The stop reason is recorded with the spend because the two are the same
        fact: a brief that used six of its six turns is not a brief that used
        six turns, it is a brief that was stopped. The trajectory goes here too,
        as one attribute, so a collector reading only the trace attributes sees
        the order the model reached for things in without walking the spans.
        """
        span = self._trace.span(BUDGET)
        span.record(
            **budget.usage(),
            stopped_by=stopped or None,
            tool_trajectory=list(self._trajectory),
            evidence_selected=[
                {
                    "evidence_id": item.evidence_id,
                    "label": item.label,
                    "document_id": item.document_id,
                    "rank": item.rank,
                    "page": item.page,
                    "position": item.position,
                    "read": item.read,
                }
                for item in evidence
            ],
        )
        span.end()

    def outcome(self, status: str, *, reason: str = "", turns: int = 0) -> None:
        """Record how the brief ended, so the record opens with its verdict."""
        self.identify(outcome=status, turns=turns)
        if reason:
            self.identify(outcome_reason=reason)
        span = self._trace.span(OUTCOME)
        span.record(status=status, reason=reason or None, turns=turns)
        span.end()

    def finish(self) -> None:
        """Close and send the trace, once."""
        if self._finished:
            return
        self._finished = True
        if self._tracer is not None:
            self._tracer.finish(self._trace)
        else:
            self._trace.end_all()
