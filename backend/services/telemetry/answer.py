"""
One answer request, recorded.

This is the vocabulary the answer path records in, gathered in one place so the
route, the evaluator, and any future caller produce the same trace for the same
work. The shape of a trace is: what the request was about, what retrieval
returned and how it was ranked, what the model was asked for and what it cost,
what happened to the citations, and what the store was finally told.

Nothing here holds a question, an answer, or Passage text in the record. The
values are identifiers, ranks, scores, counts, durations, and the safe category
of a failure. The text is passed in only so the token counts and the price can
be computed from it, and only the counts are written down.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from services.citations import PROMPT_VERSION
from services.retrieval.base import RetrievalResult
from services.telemetry.cost import cost_usd
from services.telemetry.spans import Span, Trace, Tracer, error_category
from services.telemetry.usage import input_tokens, output_tokens

#: What a request can end as, as a trace records it. These are the same words
#: the terminal stream event and the evaluation outcome use, so an operator
#: reading a trace and a reader watching the stream are told the same thing.
ANSWERED = "answered"
ABSTAINED = "abstained"
REFUSED = "refused"
CANCELLED = "cancelled"
PERSISTENCE_ERROR = "persistence_error"

#: The spans one answer request contains, in the order they happen.
RETRIEVAL = "retrieval"
GENERATION = "generation"
CITATIONS = "citations"
PERSISTENCE = "persistence"


class AnswerTrace:
    """
    The trace of one answer request, and the vocabulary it is recorded in.

    A tracer with no exporter records nothing anywhere, but the calls that
    record a trace are the same ones either way, so a trace cannot describe a
    different request than the one the application served.
    """

    def __init__(self, tracer: Tracer | None, trace: Trace, model: Any = None) -> None:
        """Record this request's steps into this trace."""
        self._tracer = tracer
        self._trace = trace
        self._model = model
        self._finished = False
        self._clock: Callable[[], float] = time.monotonic
        self._started_at: float | None = None
        self._first_token: float | None = None
        self._total: float | None = None
        #: The spans that measured a model call, so the request's timings can be
        #: written onto the step a reader actually waited for.
        self._generations: list[Span] = []

    def begin(self, clock: Callable[[], float]) -> None:
        """
        Begin the request's own clock, before the first byte goes out.

        Latency a reader feels is measured from the moment the request started,
        not from the moment one step inside it began, so a slow retrieval does
        not disappear out of the time the reader spent waiting.
        """
        self._clock = clock
        self._started_at = clock()
        self._first_token = None

    def mark_first_token(self) -> None:
        """Note that the reader has their first word, if they have not had one."""
        if self._first_token is None:
            self._first_token = self.elapsed()

    @property
    def first_token_seconds(self) -> float | None:
        """Return how long the reader waited for the first word, if any came."""
        return None if self._first_token is None else round(self._first_token, 6)

    def elapsed(self) -> float:
        """
        Return how long this request has run, in seconds.

        Once the request has ended this is the one measurement, not a fresh
        reading: a second read would report a longer wait than the request
        actually took, and a run that measured the same stream twice would
        report two different numbers for it.
        """
        if self._total is not None:
            return self._total
        if self._started_at is None:
            return 0.0
        return max(0.0, self._clock() - self._started_at)

    @classmethod
    def start(
        cls, tracer: Tracer | None, *, model: Any = None, name: str = "answer.request"
    ) -> "AnswerTrace":
        """Open a trace for one request, naming the model that will answer it."""
        trace = tracer.start(name) if tracer is not None else Trace(name=name)
        answer_trace = cls(tracer, trace, model)
        if model is not None:
            answer_trace.identify(
                provider=model.provider,
                model=model.id,
                prompt_version=PROMPT_VERSION,
            )
        return answer_trace

    @property
    def trace_id(self) -> str:
        """Return the identifier every span of this request shares."""
        return self._trace.trace_id

    @property
    def trace(self) -> Trace:
        """Return the underlying trace, for callers that read it directly."""
        return self._trace

    def identify(self, **attributes: Any) -> None:
        """Record what this request is about."""
        self._trace.identify(**attributes)

    def refused(self, category: str, status: int) -> None:
        """Record a question that was refused before any work was done."""
        self.identify(outcome=REFUSED, refusal_status=status)
        if category:
            self.identify(refusal_category=category)
        self.finish()

    @contextmanager
    def retrieval(
        self,
        *,
        document_id: str | None,
        index_generation: int | None,
        expansion_method: str = "",
    ) -> Iterator[Span]:
        """
        Open the span retrieval is recorded into.

        What came back is recorded once the query has run, so a retrieval that
        failed is still a span an operator can see rather than a gap. The filter
        is recorded as the Document's own identifiers rather than its filename:
        the filename is what a reader recognises their library by, and the
        identifiers are what two requests are correlated by.
        """
        span = self._trace.span(RETRIEVAL)
        span.record(
            document_id=document_id,
            index_generation=index_generation,
            filter={"document_id": document_id, "index_generation": index_generation},
            query_expansion_method=expansion_method,
        )
        try:
            yield span
        except Exception as error:  # noqa: BLE001 - the failure is the record
            span.fail(error_category(error))
            span.record(retrieval_outcome="error")
            raise
        finally:
            span.end()

    @contextmanager
    def generation(
        self, *, query: str, context: str, prior_turns: str
    ) -> Iterator[Span]:
        """
        Open the span one model call is recorded into.

        The text is only here to be counted. The call's outcome is recorded
        afterwards with :meth:`generated`, and the span carries the counts, the
        timings, the finish reason, and the price — never the words.
        """
        span = self._trace.span(GENERATION)
        self._generations.append(span)
        try:
            yield span
        except Exception as error:  # noqa: BLE001 - the failure is the record
            span.fail(error_category(error))
            raise
        finally:
            span.end()

    def generated(
        self,
        span: Span,
        *,
        query: str,
        context: str,
        prior_turns: str,
        generated: str | None,
        finish_reason: str | None,
        attempts: int,
        truncated: bool,
    ) -> None:
        """Record what one model call wrote, and what it cost."""
        read = input_tokens(query, context, prior_turns)
        written = None if generated is None else output_tokens(generated)
        attributes: dict[str, Any] = {
            "prompt_version": PROMPT_VERSION,
            "input_tokens": read,
            "output_tokens": written,
            "cost_usd": cost_usd(read, written, self._model) if self._model else None,
            "finish_reason": finish_reason,
            "finish_reasons": [finish_reason] if finish_reason else [],
            "truncated": truncated,
            "attempts": attempts,
            # The usage numbers under the GenAI semantic convention names, so a
            # collector that reads those names finds them without a mapping.
            "gen_ai.usage.input_tokens": read,
            "gen_ai.usage.output_tokens": written,
            "gen_ai.response.finish_reasons": (
                [finish_reason] if finish_reason else []
            ),
        }
        if self._model is not None:
            attributes["provider"] = self._model.provider
            attributes["model"] = self._model.id
            # The same facts under the GenAI semantic convention names, so a
            # collector that reads those names finds them without a mapping.
            attributes["gen_ai.system"] = self._model.provider
            attributes["gen_ai.request.model"] = self._model.id
        span.record(**attributes)

    def timings(self) -> None:
        """
        Write the request's own latencies onto the trace and its model call.

        Both are read once, when the request ends, so the number a reader felt
        and the number a collector shows cannot be two different measurements of
        the same wait.
        """
        self._total = round(self.elapsed(), 6)
        total = self._total
        first = self.first_token_seconds
        self.identify(total_latency_seconds=total, time_to_first_token_seconds=first)
        for span in self._generations:
            span.record(total_latency_ms=round(total * 1000, 3))
            if first is not None:
                span.record(time_to_first_token_ms=round(first * 1000, 3))

    def citations(
        self,
        *,
        claims: int,
        invalid_ids: int,
        grounded: bool,
        repaired: bool,
    ) -> None:
        """Record what checking the model's citations decided."""
        span = self._trace.span(CITATIONS)
        span.record(
            # Named as counts rather than as the claims themselves: the trace
            # reports how many claims survived, never what they said.
            claim_count=claims,
            invalid_source_ids=invalid_ids,
            grounded=grounded,
            repaired=repaired,
        )
        span.end()

    def persistence(self, outcome: str, *, turn_id: str, reason: str = "") -> None:
        """Record what the store was told, and whether it took it."""
        span = self._trace.span(PERSISTENCE)
        span.record(turn_id=turn_id, outcome=outcome)
        if reason:
            span.record(reason=reason)
        span.end()
        self.identify(outcome=outcome)

    def fail(self, span_name: str, category: str) -> None:
        """Record which kind of failure ended one step of this request."""
        span = self._trace.first(span_name)
        if span is not None and span.error_category is None:
            span.fail(category)

    def finish(self, outcome: str = "") -> None:
        """Close and send the trace, once."""
        if self._finished:
            return
        self._finished = True
        if outcome:
            self.identify(outcome=outcome)
        if self._tracer is not None:
            self._tracer.finish(self._trace)
        else:
            self._trace.end_all()


def retrieval_attributes(result: RetrievalResult) -> dict[str, Any]:
    """
    Return what one retrieval produced, as a trace records it.

    Ranks, scores, hashes, and Pages are the whole record: they are what an
    operator compares between two runs, and none of them is the Passage itself.
    """
    return {
        "retrieval_method": result.method,
        "retrieval_outcome": result.outcome,
        "candidate_count": len(result.candidates),
        "selected_count": len(result.sources),
        "candidates": [candidate.to_dict() for candidate in result.candidates],
        "rerank": result.rerank or {"applied": False},
    }
