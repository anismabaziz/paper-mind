"""
One record of one step of one answer request.

A span is what an operator reads: what the step was, how long it took, what it
decided, and — when it failed — which kind of failure it was. It carries
identifiers, ranks, scores, counts, and durations rather than text, so a trace
can be read from a file or shipped to a collector without becoming a second
copy of the Document.

The nesting is one answer request. The trace is the request; retrieval,
generation, citation validation, and persistence are spans inside it, so a
trace can be read top to bottom in the order the work happened, and every span
carries the trace's correlation identifiers so one that reaches a collector on
its own still names the Document, Conversation, Turn, and index generation it
belongs to.

A failure is recorded by kind and never by message. A provider's exception text
can quote the prompt, the question, or the key it was given, and an operator
reading a trace needs none of that.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, Sequence


def new_trace_id() -> str:
    """Return one identifier for a trace."""
    return uuid.uuid4().hex


def error_category(error: BaseException) -> str:
    """
    Return the safe category one failure is recorded as.

    The typed failures the answer path already distinguishes are named by the
    same category the reader's terminal event carries, so a trace and the
    stream never tell two operators two different stories. Anything else is
    named by its class, which describes the failure without quoting it.
    """
    from services.llm.base import EmptyAnswerError, ProviderTimeoutError
    from services.retrieval.base import (
        VectorDimensionError,
        VectorStoreConfigurationError,
        VectorStoreUnavailableError,
    )

    for failure, category in (
        (ProviderTimeoutError, "timeout"),
        (EmptyAnswerError, "empty_output"),
        (VectorStoreUnavailableError, "vector_store_unavailable"),
        (VectorStoreConfigurationError, "vector_store_configuration"),
        (VectorDimensionError, "vector_dimension_mismatch"),
    ):
        if isinstance(error, failure):
            return category
    return type(error).__name__


@dataclass
class Span:
    """One step of a request: what it decided, how long it took, how it ended."""

    name: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    #: Set when the step failed, to the safe category rather than the message.
    error_category: str | None = None
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def record(self, **attributes: Any) -> None:
        """Record what this step decided."""
        self.attributes.update(attributes)

    def fail(self, category: str) -> None:
        """Record which kind of failure ended this step."""
        self.error_category = category

    def end(self) -> None:
        """Close the span, keeping the first end time if it is closed twice."""
        if self.ended_at is None:
            self.ended_at = time.time()

    @property
    def duration_ms(self) -> float | None:
        """Return how long the step took, once it has ended."""
        if self.ended_at is None:
            return None
        return round((self.ended_at - self.started_at) * 1000, 3)

    def to_dict(self) -> dict[str, Any]:
        """Return the span as a trace record and an exporter write it."""
        return {
            "name": self.name,
            "span_id": self.span_id,
            "error_category": self.error_category,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "attributes": dict(self.attributes),
        }


class SpanExporter(Protocol):
    """Where finished traces go."""

    def export(self, trace: "Trace") -> None:
        """Send one finished trace, or drop it if the destination is unavailable."""
        ...


@dataclass
class Trace:
    """
    Everything one answer request did, in the order it did it.

    The correlation identifiers are held here and copied onto every span, so a
    span that reaches a collector on its own still names what it was about.
    """

    name: str = "answer.request"
    trace_id: str = field(default_factory=new_trace_id)
    attributes: dict[str, Any] = field(default_factory=dict)
    spans: list[Span] = field(default_factory=list)

    def identify(self, **attributes: Any) -> None:
        """Record what this request is about, before any step runs."""
        self.attributes.update(attributes)

    def span(self, name: str) -> Span:
        """Open a span of this trace."""
        span = Span(name=name)
        self.spans.append(span)
        return span

    def find(self, name: str) -> list[Span]:
        """Return every span of this trace with this name, in order."""
        return [span for span in self.spans if span.name == name]

    def first(self, name: str) -> Span | None:
        """Return the first span with this name, or None when there is none."""
        found = self.find(name)
        return found[0] if found else None

    def end_all(self) -> None:
        """Close whatever is still open, so an abandoned request is still readable."""
        for span in self.spans:
            span.end()

    def copy(self) -> "Trace":
        """Return an independent copy, so one destination cannot redact another."""
        return Trace(
            name=self.name,
            trace_id=self.trace_id,
            attributes=dict(self.attributes),
            spans=[
                Span(
                    name=span.name,
                    span_id=span.span_id,
                    error_category=span.error_category,
                    started_at=span.started_at,
                    ended_at=span.ended_at,
                    attributes=span.attributes,
                )
                for span in self.spans
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the trace as a record and an exporter write it."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "attributes": dict(self.attributes),
            "spans": [span.to_dict() for span in self.spans],
        }


@dataclass(frozen=True)
class Sink:
    """
    One destination and what that destination is allowed to record.

    The redactor lives with the exporter rather than on the tracer, because
    what a destination may say is a property of the destination. A local file
    under an active, time-bounded capture window may hold the question and the
    evidence; a collector gets fingerprints whatever the window is doing.
    """

    exporter: SpanExporter
    redactor: Any = None


class Tracer:
    """
    Creates traces and hands the finished ones to each destination.

    Tracing is not allowed to change what the application does, so sending a
    trace is best effort: a destination that is down, slow, or refusing a write
    is dropped and the request carries on. A tracer with no sinks still builds
    the trace, so the code that records one is the same whether or not
    anybody is reading it.
    """

    def __init__(self, sinks: Sequence[Sink] = ()) -> None:
        """Send finished traces to these destinations."""
        self._sinks = list(sinks)

    @property
    def enabled(self) -> bool:
        """Report whether finished traces are recorded anywhere."""
        return bool(self._sinks)

    def start(self, name: str = "answer.request") -> Trace:
        """Open a trace for one request."""
        return Trace(name=name)

    def finish(self, trace: Trace) -> Trace:
        """Close every span and send the trace to each destination."""
        trace.end_all()
        for sink in self._sinks:
            outgoing = self._redacted(trace, sink)
            try:
                sink.exporter.export(outgoing)
            except Exception:  # noqa: BLE001 - a sink must not fail a request
                continue
        return trace

    @staticmethod
    def _redacted(trace: Trace, sink: Sink) -> Trace:
        """Return the trace as this destination is allowed to receive it."""
        outgoing = trace.copy()
        if sink.redactor is None:
            return outgoing
        outgoing.attributes = sink.redactor.scrub(outgoing.attributes)
        for span in outgoing.spans:
            span.attributes = sink.redactor.scrub(span.attributes)
        return outgoing

    @contextmanager
    def step(self, trace: Trace, name: str) -> Iterator[Span]:
        """Run one step of a trace, closing and classifying it however it ends."""
        span = trace.span(name)
        try:
            yield span
        except Exception as error:  # noqa: BLE001 - the failure is the record
            span.fail(error_category(error))
            raise
        finally:
            span.end()
