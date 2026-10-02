"""
The answer path: retrieve, ground, generate, validate, persist.

One question asked about one Document travels this path in the HTTP route and
in the evaluator. Both call this service, so a retrieval or generation result
that a report describes is the one the application produced, not a second
implementation of it.

The path has two phases because the answer is streamed. Everything decided
before the first byte — is this Document answerable at all, what evidence
exists, is there enough of it to answer, has the question been committed —
happens in :meth:`AnswerService.resolve`, so a refusal is a refusal rather
than a stream that opens and immediately fails. Generation, citation
validation, and the terminal outcome happen in :meth:`AnswerService.stream`,
which yields named events; the route turns those into server-sent events and
the evaluator reads them as case outcomes.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

from errors import error_payload
from services.abstention import Abstention, abstention_for
from services.accounts.chat_settings_service import ModelCapabilities
from services.chat_context import build_chat_context, build_model_rewriter
from services.citations import (
    PROMPT_VERSION,
    UNRESOLVED_CITATIONS_REASON,
    AnswerSplitter,
    ValidatedCitations,
    assign_source_ids,
    parse_claims,
    prune_conflicting_claims,
    render_evidence,
    repair_instruction,
    validate_claims,
)
from services.deletion import deletion_block_payload, is_deleting_record
from services.indexing.manifest import manifest_from_json
from services.indexing.readiness import (
    CHAT_REFUSAL_MESSAGES,
    UNREADABLE_DELETING,
    UNREADABLE_STALE,
    document_refusal,
)
from services.indexing.state import index_status
from services.llm.base import EmptyAnswerError, LLMProvider, ProviderTimeoutError
from services.retrieval.base import (
    RetrievalMethod,
    RetrievalOutcome,
    RetrievalResult,
    VectorDimensionError,
    VectorStoreConfigurationError,
    VectorStoreUnavailableError,
)
from services.retrieval.query_expansion import expand_query
from services.telemetry.answer import (
    ABSTAINED,
    ANSWERED,
    CANCELLED,
    CITATIONS,
    GENERATION,
    PERSISTENCE,
    PERSISTENCE_ERROR,
    AnswerTrace,
    retrieval_attributes,
)
from services.telemetry.factory import tracer_for
from services.telemetry.spans import error_category
from settings import Settings

log = logging.getLogger(__name__)

MAX_QUERY_CHARS = 8192


@dataclass(frozen=True)
class AnswerRequest:
    """One question about one Document, with the provider that will answer it."""

    filename: str
    query: str
    provider: LLMProvider
    model: ModelCapabilities


@dataclass(frozen=True)
class Refusal:
    """
    Why a question was not answered, before any stream started.

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


@dataclass(frozen=True)
class AnswerProvenance:
    """
    What produced one answer, so two answers can be compared honestly.

    An answer is only comparable to another answer when the same index
    generation, retrieval method, prompt, provider, model, and settings
    produced both, so a run records all of it rather than a bare verdict.
    """

    index_manifest: dict[str, Any] | None
    index_generation: int | None
    index_state: str
    retrieval_method: RetrievalMethod
    retrieval_outcome: RetrievalOutcome
    prompt_version: str
    provider: str
    model: str
    settings: dict[str, Any]
    retrieval_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return the provenance as the run record a report stores."""
        return asdict(self)


@dataclass(frozen=True)
class AnswerSettings:
    """The bounded-context and retrieval settings one answer was built with."""

    recent_turns: int
    prior_turns_token_budget: int
    context_token_budget: int
    query_rewrite: bool
    max_expansion_chars: int
    rerank_enabled: bool
    rerank_model: str
    embedding_model: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    index_name: str

    @classmethod
    def from_settings(cls, settings: Settings) -> "AnswerSettings":
        """Read the answer-affecting settings out of the running configuration."""
        return cls(
            recent_turns=settings.query_context.recent_turns,
            prior_turns_token_budget=settings.query_context.prior_turns_token_budget,
            context_token_budget=settings.query_context.context_token_budget,
            query_rewrite=settings.query_context.query_rewrite,
            max_expansion_chars=settings.query_context.max_expansion_chars,
            rerank_enabled=settings.rerank.enabled,
            rerank_model=settings.rerank.rerank_model,
            embedding_model=settings.embedding.embedding_model,
            chunk_size_tokens=settings.chunking.chunk_size_tokens,
            chunk_overlap_tokens=settings.chunking.chunk_overlap_tokens,
            index_name=settings.vector.index_name,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the settings as a run record stores them."""
        return asdict(self)


@dataclass(frozen=True)
class Retrieval:
    """
    What retrieval produced for one question, and how long it took.

    The payload is what the stream and the reader see; it carries no timings,
    because latency belongs in the run record rather than in an answer's
    evidence.
    """

    payload: dict[str, Any]
    seconds: float


@dataclass(frozen=True)
class ResolvedTurn:
    """A question that reached the model, with everything it was answered with."""

    turn_id: str
    request: AnswerRequest
    context: str
    prior_turns: str
    sources: list[dict[str, Any]]
    retrieval: Retrieval
    abstention: Abstention | None
    provenance: AnswerProvenance
    #: The trace of this request, carried from the decision phase so the
    #: outcome, the citations, and the store write land in the same trace the
    #: retrieval that led to them was recorded in.
    trace: AnswerTrace


@dataclass(frozen=True)
class AnswerFailure:
    """What one way an answer can fail shows the user and records on the Turn."""

    message: str
    reason: str
    category: str


#: Each way an answer can end before it is stored, with the words the reader
#: gets and the reason the Turn is closed with.
PROVIDER_DOWN = AnswerFailure(
    "Sorry. The language model is unavailable right now. Please try again.",
    "provider failure",
    "provider",
)
PROVIDER_SLOW = AnswerFailure(
    "This answer took too long and was stopped. Try asking a narrower question.",
    "provider timeout",
    "timeout",
)
PROVIDER_SILENT = AnswerFailure(
    "The model returned nothing for this question. Try rephrasing it.",
    "empty answer",
    "empty_output",
)
UNSAVED = AnswerFailure(
    "The answer could not be saved. Please ask the question again.",
    "answer could not be saved",
    "persistence",
)
UNCITED = AnswerFailure(
    "The answer cited a passage that was not supplied with the question, "
    "so it could not be shown. Please ask the question again.",
    UNRESOLVED_CITATIONS_REASON,
    "citations",
)

#: Which step of a trace a terminal event blames for the failure. The event
#: names the outcome a reader sees; this names the step that caused it, so a
#: provider timeout is recorded against generation and a failed save against
#: persistence rather than both against whichever step happened to be open.
_SPAN_FOR_EVENT = {
    "provider_error": GENERATION,
    "citation_error": CITATIONS,
    "persistence_error": PERSISTENCE,
}


def vector_store_refusal(error: Exception) -> Refusal:
    """Return the refusal a vector-store failure produces."""
    if isinstance(error, VectorStoreUnavailableError):
        return Refusal(
            503,
            "vector_store_unavailable",
            "Vector store is unavailable",
        )
    if isinstance(error, VectorStoreConfigurationError):
        return Refusal(
            500,
            "vector_store_configuration",
            "Vector store configuration is invalid",
        )
    raise TypeError("Unsupported vector-store error")


def _normalize_source(source: dict[str, Any]) -> dict[str, Any]:
    """Return one retrieved Passage with the page key the app reads."""
    page = source["page"] if "page" in source else source.get("page_no")
    return {**source, "page": page if page is not None else None}


@dataclass(frozen=True)
class ResolvedAnswer:
    """One generated answer: the prose to show, and the claims that survived."""

    answer: str
    citations: ValidatedCitations


class AnswerService:
    """
    Answers questions about stored Documents.

    The service owns the document's index state, the Conversation, retrieval,
    the bounded context, abstention, generation, citation validation, and the
    Turn outcomes. The route supplies the provider and the HTTP shapes; the
    evaluator supplies the same two and reads the events back as outcomes.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        repositories: Any,
        embedding_service: Any,
        vector_service: Any,
        tracer: Any = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """
        Bind the answer path to the dependencies the app serves with.

        ``tracer`` is where each request's trace goes. Left out, the answer path
        builds the one the running configuration asks for, so a deployment
        configures observability in settings and not in each caller.
        """
        self._settings = settings
        self._repositories = repositories
        self._embeddings = embedding_service
        self._vectors = vector_service
        self._tracer = tracer if tracer is not None else tracer_for(settings)
        self._clock = clock

    @property
    def conversations(self) -> Any:
        """Return the conversation store answers are recorded in."""
        return self._repositories.conversations

    @property
    def files(self) -> Any:
        """Return the document store the index state is judged from."""
        return self._repositories.files

    def resolve(self, request: AnswerRequest) -> ResolvedTurn | Refusal:
        """
        Decide everything an answer needs before the model is called.

        Returns a :class:`Refusal` when the question cannot be asked at all —
        an unknown Document, one being deleted or reindexed, an index that no
        longer matches the running configuration, a missing Conversation, or a
        question the app will not accept. Everything else returns a
        :class:`ResolvedTurn` with the question already committed, so the
        reader can see it in history whatever the answer turns out to be.

        Every request opens a trace, including one that is refused: a question
        an operator cannot account for is exactly the question a trace is for.
        """
        trace = AnswerTrace.start(self._tracer, model=request.model)
        # The question is recorded so the trace can show there was one; the
        # redactor replaces it with a fingerprint unless a local capture window
        # is open.
        trace.identify(filename=request.filename, query=request.query)
        refusal = self._refuse_unanswerable(request)
        if refusal is not None:
            return self._refused(trace, refusal)

        file_record = self.files.get_file(request.filename)
        conversation_id = self.conversations.get_conversation_id(file_record["id"])
        if not conversation_id:
            return self._refused(
                trace, Refusal(404, "conversation_not_found", "Conversation not found")
            )
        if not isinstance(request.query, str) or not request.query.strip():
            return self._refused(
                trace,
                Refusal(400, "invalid_query", "Query and Filename are required"),
            )
        if len(request.query) > MAX_QUERY_CHARS:
            return self._refused(
                trace, Refusal(400, "query_too_long", "Query is too long")
            )

        trace.identify(
            document_id=file_record["id"],
            conversation_id=conversation_id,
            index_generation=file_record.get("index_generation"),
        )
        try:
            retrieval, sources, context, prior_turns_text, result = (
                self._retrieve_context(request, file_record, conversation_id, trace)
            )
        except (
            VectorStoreUnavailableError,
            VectorStoreConfigurationError,
        ) as exc:
            log.exception("/response vector store failed for %s", request.filename)
            # The retrieval span already carries the failure's category: the
            # context it ran in classifies whatever escapes it.
            return self._refused(trace, vector_store_refusal(exc))
        except VectorDimensionError as exc:
            log.warning(
                "/response vector dimension mismatch for %s: %s", request.filename, exc
            )
            return self._refused(
                trace, Refusal(409, "vector_dimension_mismatch", str(exc))
            )
        except Exception:
            log.exception("/response retrieval failed for %s", request.filename)
            return self._refused(trace, Refusal(500, "", "Internal server error"))

        if not self._still_present(request.filename):
            log.warning(
                "/response refusing chat for deleting document %s", request.filename
            )
            return self._refused(
                trace, deletion_block_refusal(self.files.get_file(request.filename))
            )

        # Decided before the Turn is committed, and acted on below: whether
        # this question gets a model call is settled the moment retrieval is.
        abstention = abstention_for(sources)
        try:
            turn_id = self.conversations.start_turn(conversation_id, request.query)
        except Exception:
            log.exception(
                "/response could not commit the question for %s", request.filename
            )
            return self._refused(trace, Refusal(500, "", "Internal server error"))

        trace.identify(turn_id=turn_id)
        return ResolvedTurn(
            turn_id=turn_id,
            request=request,
            context=context,
            prior_turns=prior_turns_text,
            sources=sources,
            retrieval=retrieval,
            abstention=abstention,
            provenance=self._provenance(request, file_record, retrieval),
            trace=trace,
        )

    @staticmethod
    def _refused(trace: AnswerTrace, refusal: Refusal) -> Refusal:
        """Record a refused request and return the refusal unchanged."""
        trace.refused(refusal.category, refusal.status)
        return refusal

    def stream(self, resolved: ResolvedTurn) -> Iterator[AnswerEvent]:
        """
        Yield what happens to one resolved question, ending on one outcome.

        The turn is already on record, so a stream that ends early still leaves
        a Turn the reader can see. A client that walks away mid-answer closes
        the Turn as cancelled rather than leaving it pending forever.

        The clock is read once here, before the first byte, and everything the
        trace reports about this request is measured against it: the time to
        the first token and the whole exchange are both what a reader waited,
        not what one step inside the request cost. The trace is closed here
        rather than in ``resolve`` because the work it records is this — the
        retrieval span is already open, and the outcome, the citations, and the
        store write have not happened yet.
        """
        resolved.trace.begin(clock=self._clock)
        try:
            if resolved.abstention is not None:
                yield from self._abstain(resolved, resolved.abstention)
                return
            yield from self._generate(resolved)
        finally:
            resolved.trace.timings()
            resolved.trace.finish()

    def _refuse_unanswerable(self, request: AnswerRequest) -> Refusal | None:
        """
        Return the refusal that stops a question before retrieval.

        Whether a Document can be read is judged in one place for every path
        that reads Documents; what is added here is how a question about it
        reads, since that is what this route's reader is waiting on.
        """
        unreadable = document_refusal(
            self._repositories, self._settings, request.filename
        )
        if unreadable is None:
            return None
        if unreadable.category == UNREADABLE_DELETING:
            # The deletion payload names the retry as well as the block, so it
            # speaks for itself rather than being reworded here.
            return deletion_block_refusal(self.files.get_file(request.filename))
        if unreadable.category == UNREADABLE_STALE:
            index = unreadable.detail.get("index") or {}
            log.info(
                "/response refused for stale index %s: %s",
                request.filename,
                index.get("changes"),
            )
        status, message = CHAT_REFUSAL_MESSAGES[unreadable.category]
        return Refusal(
            status,
            unreadable.category,
            message,
            unreadable.detail,
        )

    def _retrieve_context(
        self,
        request: AnswerRequest,
        file_record: dict[str, Any],
        conversation_id: str,
        trace: AnswerTrace,
    ) -> tuple[Retrieval, list[dict[str, Any]], str, str, RetrievalResult]:
        """
        Return what retrieval produced, the supplied Passages, and the prompt.

        The evidence, the transcript, and the question are bounded the same way
        the reader's request is, so what the model is shown here is what it was
        shown in production — including which Passages the budget dropped.
        """
        limits = self._settings.query_context
        recent_turns = self.conversations.get_recent_turns(
            conversation_id, limits.recent_turns
        )
        turns_in_conversation = self.conversations.count_answered_turns(conversation_id)
        # Two different windows, deliberately: expansion takes the recent
        # questions up to the Turn limit, capped by max_expansion_chars; the
        # transcript sent to the model is bounded by prior_turns_token_budget.
        prior_questions = [
            turn["question"] for turn in recent_turns if turn.get("question")
        ]
        rewriter = (
            build_model_rewriter(request.provider)
            if limits.query_rewrite and prior_questions
            else None
        )
        expansion = expand_query(
            request.query,
            prior_questions,
            max_chars=limits.max_expansion_chars,
            rewrite=rewriter,
        )
        started = self._clock()
        with trace.retrieval(
            document_id=file_record.get("id"),
            index_generation=file_record.get("index_generation"),
            expansion_method=expansion.method,
        ) as span:
            query_embedding = self._embeddings.embed_texts(expansion.expanded_query)[0]
            retrieval_result = self._vectors.query_vectors(
                query_embedding,
                request.filename,
                query_text=expansion.expanded_query,
                generation=file_record.get("index_generation"),
                include_legacy=file_record.get("index_generation") is None,
            )
            elapsed = self._clock() - started
            sources = assign_source_ids(
                [_normalize_source(source) for source in retrieval_result.sources]
            )
            chat_context = build_chat_context(
                request.query,
                sources,
                recent_turns,
                expansion,
                max_turns=limits.recent_turns,
                prior_turns_token_budget=limits.prior_turns_token_budget,
                context_token_budget=limits.context_token_budget,
                turns_in_conversation=turns_in_conversation,
                input_token_budget=request.model.max_input_tokens,
            )
            span.record(
                **retrieval_attributes(retrieval_result),
                latency_ms=round(elapsed * 1000, 3),
                dropped_turns=chat_context.dropped_turns,
                dropped_sources=chat_context.dropped_sources,
            )
        # Citations must match what the model actually saw, so a Citation
        # Source the budget dropped is not stored against the answer.
        kept_sources = list(chat_context.sources)
        retrieval = Retrieval(
            payload={
                "method": retrieval_result.method,
                "outcome": retrieval_result.outcome,
                **expansion.to_dict(),
                "dropped_turns": chat_context.dropped_turns,
                "dropped_sources": chat_context.dropped_sources,
            },
            seconds=elapsed,
        )
        return (
            retrieval,
            kept_sources,
            chat_context.context,
            chat_context.prior_turns,
            retrieval_result,
        )

    def _provenance(
        self,
        request: AnswerRequest,
        file_record: dict[str, Any],
        retrieval: Retrieval,
    ) -> AnswerProvenance:
        """Record the index, prompt, provider, model, and settings behind a run."""
        stored = manifest_from_json(file_record.get("index_manifest"))
        return AnswerProvenance(
            index_manifest=stored.to_dict() if stored else None,
            index_generation=file_record.get("index_generation"),
            index_state=index_status(self.files, file_record, self._settings).state,
            retrieval_method=retrieval.payload["method"],
            retrieval_outcome=retrieval.payload["outcome"],
            prompt_version=PROMPT_VERSION,
            # The catalog record names the provider and model that answered, so
            # a run is identified by the configuration rather than by whichever
            # SDK class happened to serve it.
            provider=request.model.provider,
            model=request.model.id,
            settings=AnswerSettings.from_settings(self._settings).to_dict(),
            retrieval_seconds=retrieval.seconds,
        )

    def _still_present(self, filename: str) -> bool:
        """Return whether the Document survived a concurrent deletion."""
        try:
            current = self.files.get_file(filename)
        except Exception:
            log.exception("/response deletion race check failed for %s", filename)
            return True
        if current is None:
            return False
        return not is_deleting_record(current)

    def _record_outcome(self, filename: str, record: Callable[..., Any], *args: Any):
        """Apply one terminal Turn outcome unless the Document is gone."""
        try:
            if not self._still_present(filename):
                log.warning(
                    "/response skipping turn outcome after concurrent delete for %s",
                    filename,
                )
                return
            record(*args)
        except Exception:
            log.exception("/response could not record turn outcome for %s", filename)

    def _fail(
        self, resolved: ResolvedTurn, failure: AnswerFailure, event_name: str
    ) -> Iterator[AnswerEvent]:
        """
        Close the Turn as failed and yield the terminal event.

        The failure the reader is told and the failure the trace records are the
        same one: the category on the terminal event is the one the step that
        failed is recorded as, so the stream and the trace never disagree about
        why an answer ended.
        """
        resolved.trace.fail(_SPAN_FOR_EVENT[event_name], failure.category)
        resolved.trace.identify(outcome=event_name)
        self._record_outcome(
            resolved.request.filename,
            self.conversations.fail_turn,
            resolved.turn_id,
            failure.message,
            failure.reason,
        )
        yield AnswerEvent(
            event_name,
            error_payload(failure.category, failure.message),
        )

    def _abstain(
        self, resolved: ResolvedTurn, abstention: Abstention
    ) -> Iterator[AnswerEvent]:
        """
        Record an abstention and yield it as the turn's only outcome.

        Nothing is generated, and there is no Passage to cite, so the answer
        carries no sources: the reader is told the question was not answerable
        from this Document rather than handed prose that looks like one.
        """
        log.info(
            "/response abstained for %s: %s (%s retrieved, %s usable)",
            resolved.request.filename,
            abstention.reason,
            abstention.retrieved,
            abstention.usable,
        )
        try:
            recorded = self.conversations.abstain_turn(
                resolved.turn_id, abstention.message, abstention.reason
            )
        except Exception as exc:
            log.exception(
                "/response could not record the abstention for %s",
                resolved.request.filename,
            )
            resolved.trace.fail(PERSISTENCE, error_category(exc))
            resolved.trace.persistence(PERSISTENCE_ERROR, turn_id=resolved.turn_id)
            yield AnswerEvent(
                "persistence_error",
                error_payload(UNSAVED.category, UNSAVED.message),
            )
            return
        if not recorded:
            log.warning(
                "/response turn %s already ended for %s",
                resolved.turn_id,
                resolved.request.filename,
            )
            resolved.trace.persistence(
                PERSISTENCE_ERROR,
                turn_id=resolved.turn_id,
                reason="turn already ended",
            )
            yield AnswerEvent(
                "persistence_error",
                error_payload(UNSAVED.category, UNSAVED.message),
            )
            return
        resolved.trace.identify(abstention_reason=abstention.reason)
        resolved.trace.persistence(ABSTAINED, turn_id=resolved.turn_id)
        yield AnswerEvent("start", {"turn_id": resolved.turn_id})
        yield AnswerEvent(
            "abstained",
            {**abstention.to_dict(), "retrieval": resolved.retrieval.payload},
        )

    def _check_citations(
        self, resolved: ResolvedTurn, generated: str
    ) -> ResolvedAnswer | None:
        """
        Read one generated answer's claims and check every citation in them.

        Returns the answer prose beside its validated claims, or None when a
        citation still names nothing supplied after the one repair the app is
        willing to spend. A citation to a Passage the model was never shown is
        false, and a false citation that reaches the reader is worse than an
        answer that admits it could not be shown.
        """
        evidence = resolved.sources
        allowed = [source["source_id"] for source in evidence]
        parsed = parse_claims(generated)
        citations = validate_claims(
            prune_conflicting_claims(parsed.answer, parsed.claims), allowed
        )
        if not citations.invalid_ids:
            resolved.trace.citations(
                claims=len(citations.claims),
                invalid_ids=0,
                grounded=citations.grounded,
                repaired=False,
            )
            return ResolvedAnswer(parsed.answer, citations)
        log.warning(
            "/response cited passages not supplied for %s: %s of %s",
            resolved.request.filename,
            ",".join(citations.invalid_ids),
            ",".join(allowed),
        )
        # One repair, and only the mapping: the reader has already read the
        # answer, so a second version of the same sentences is a different
        # answer. A repair that cannot be read is as false as the original.
        repaired = parse_claims(
            resolved.request.provider.generate_response(
                repair_instruction(parsed.answer, parsed.claims, allowed),
                render_evidence(evidence),
            )
        )
        repaired_claims = prune_conflicting_claims(parsed.answer, repaired.claims)
        citations = validate_claims(repaired_claims, allowed)
        if repaired_claims and not citations.invalid_ids:
            resolved.trace.citations(
                claims=len(citations.claims),
                invalid_ids=0,
                grounded=citations.grounded,
                repaired=True,
            )
            return ResolvedAnswer(parsed.answer, citations)
        log.warning(
            "/response citations unresolved for %s: %s",
            resolved.request.filename,
            ",".join(citations.invalid_ids),
        )
        resolved.trace.citations(
            claims=len(citations.claims),
            invalid_ids=len(citations.invalid_ids),
            grounded=citations.grounded,
            repaired=True,
        )
        return None

    def _done(
        self, resolved: ResolvedTurn, answer: ResolvedAnswer, stored: str
    ) -> AnswerEvent:
        """
        Build the terminal event for a stream that stored an answer.

        Only ever yielded after ``complete_turn`` committed, so a reader that
        sees it knows the answer, its claims, and its Citation Sources are all
        in history.
        """
        model = resolved.request.model
        reason = getattr(resolved.request.provider, "last_finish_reason", None)
        # No finish reason at all means the provider never said the answer was
        # over — the app's own cap stopped it, or the stream ended early. That
        # is not proof the answer is whole, so it is reported as unproven
        # rather than as complete.
        truncated = reason not in model.complete_finish_reasons
        return AnswerEvent(
            "done",
            {
                "done": True,
                # The answer as it was stored, so the terminal event describes
                # the whole exchange rather than only its evidence.
                "answer": stored,
                "sources": resolved.sources,
                **answer.citations.to_dict(),
                "retrieval": resolved.retrieval.payload,
                "finish_reason": reason,
                "truncated": truncated,
            },
        )

    def _generate(self, resolved: ResolvedTurn) -> Iterator[AnswerEvent]:
        """Stream one answer and end it on the outcome it actually had."""
        request = resolved.request
        yield AnswerEvent("start", {"turn_id": resolved.turn_id})
        fragments: list[str] = []
        splitter = AnswerSplitter()
        with resolved.trace.generation(
            query=request.query,
            context=resolved.context,
            prior_turns=resolved.prior_turns,
        ) as span:
            try:
                for token in request.provider.stream_response(
                    request.query, resolved.context, resolved.prior_turns
                ):
                    fragments.append(token)
                    resolved.trace.mark_first_token()
                    # The claims block is written in the model's own output but
                    # belongs to the app, so it is collected and never shown.
                    visible = splitter.feed(token)
                    if visible:
                        yield AnswerEvent("token", {"text": visible})
                trailing = splitter.finish()
                if trailing:
                    yield AnswerEvent("token", {"text": trailing})
            except GeneratorExit:
                # The reader left mid-answer. The question stays on record as
                # cancelled rather than pending, so it is never stranded, and no
                # fragment the provider still holds can reach a later request.
                self._record_outcome(
                    request.filename,
                    self.conversations.cancel_turn,
                    resolved.turn_id,
                    "client disconnected",
                )
                resolved.trace.identify(outcome=CANCELLED)
                resolved.trace.persistence(
                    CANCELLED, turn_id=resolved.turn_id, reason="client disconnected"
                )
                self._record_generation(
                    resolved,
                    span,
                    generated="".join(fragments).strip() or None,
                )
                raise
            except ProviderTimeoutError:
                log.warning("/response generation timed out for %s", request.filename)
                self._record_generation(
                    resolved,
                    span,
                    generated="".join(fragments).strip() or None,
                )
                yield from self._fail(resolved, PROVIDER_SLOW, "provider_error")
                return
            except EmptyAnswerError:
                log.warning("/response generation was empty for %s", request.filename)
                self._record_generation(
                    resolved,
                    span,
                    generated=None,
                )
                yield from self._fail(resolved, PROVIDER_SILENT, "provider_error")
                return
            except Exception as exc:  # noqa: BLE001 - one provider outcome
                log.error(
                    "/response generation failed for %s: %s",
                    request.filename,
                    type(exc).__name__,
                )
                self._record_generation(
                    resolved,
                    span,
                    generated="".join(fragments).strip() or None,
                )
                yield from self._fail(resolved, PROVIDER_DOWN, "provider_error")
                return

            answer = "".join(fragments).strip()
            self._record_generation(resolved, span, generated=answer or None)

        if not self._still_present(request.filename):
            # The Document and its Conversation went away mid-answer, so there
            # is nothing left to store this in. Saying the answer was saved
            # would be a lie the reader would replay from history.
            log.warning(
                "/response abandoning answer for deleted document %s", request.filename
            )
            try:
                self.conversations.cancel_turn(resolved.turn_id, "document deleted")
            except Exception as exc:
                log.exception(
                    "/response could not cancel turn for deleted document %s",
                    request.filename,
                )
                resolved.trace.fail(PERSISTENCE, error_category(exc))
            resolved.trace.identify(outcome=CANCELLED)
            resolved.trace.persistence(
                CANCELLED, turn_id=resolved.turn_id, reason="document deleted"
            )
            yield AnswerEvent("cancelled", {"reason": "document deleted"})
            return
        citations = self._check_citations(resolved, answer)
        if citations is None:
            yield from self._fail(resolved, UNCITED, "citation_error")
            return
        stored = citations.answer or answer
        try:
            completed = self.conversations.complete_turn(
                resolved.turn_id,
                stored,
                resolved.sources,
                claims=[claim.to_dict() for claim in citations.citations.claims],
            )
        except Exception as exc:
            log.exception("failed to persist answer for %s", request.filename)
            resolved.trace.persistence(PERSISTENCE_ERROR, turn_id=resolved.turn_id)
            yield from self._fail(resolved, UNSAVED, "persistence_error")
            return
        if not completed:
            # The Turn already reached a terminal state, so this answer is not
            # in history. A success event would put it on screen as if it were.
            log.warning(
                "/response turn %s already ended for %s",
                resolved.turn_id,
                request.filename,
            )
            resolved.trace.persistence(
                PERSISTENCE_ERROR,
                turn_id=resolved.turn_id,
                reason="turn already ended",
            )
            yield from self._fail(resolved, UNSAVED, "persistence_error")
            return
        resolved.trace.persistence(ANSWERED, turn_id=resolved.turn_id)
        yield self._done(resolved, citations, stored)

    def _record_generation(
        self, resolved: ResolvedTurn, span: Any, *, generated: str | None
    ) -> None:
        """Record what the one model call wrote, whether or not it answered."""
        reason = getattr(resolved.request.provider, "last_finish_reason", None)
        resolved.trace.generated(
            span,
            query=resolved.request.query,
            context=resolved.context,
            prior_turns=resolved.prior_turns,
            generated=generated,
            finish_reason=reason,
            attempts=int(getattr(resolved.request.provider, "last_attempts", 0) or 0),
            truncated=reason not in resolved.request.model.complete_finish_reasons,
        )


def deletion_block_refusal(file_record: dict[str, Any] | None) -> Refusal:
    """Return the refusal that blocks work on a Document being deleted."""
    record = file_record or {"deletion_state": "deleting"}
    payload = deletion_block_payload(record)
    return Refusal(
        409,
        payload["category"],
        payload["error"],
        {
            key: value
            for key, value in payload.items()
            if key not in ("error", "category")
        },
    )
