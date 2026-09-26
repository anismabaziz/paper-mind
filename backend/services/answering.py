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

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

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
from services.indexing.state import index_status
from services.llm.base import EmptyAnswerError, LLMProvider, ProviderTimeoutError
from services.retrieval.base import (
    RetrievalMethod,
    RetrievalOutcome,
    VectorDimensionError,
    VectorStoreConfigurationError,
    VectorStoreUnavailableError,
)
from services.retrieval.query_expansion import expand_query
from settings import Settings

log = logging.getLogger(__name__)

MAX_QUERY_CHARS = 8192

#: Job states that mean the Document's vectors are being rewritten right now.
REINDEXING_STATES = ("queued", "running", "cancelling")


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

    ``status`` and ``to_dict`` are what a client is told. ``category`` is empty
    for a failure the app cannot classify, because a guess would be worse than
    silence. The evaluator reads the same refusal as a case outcome, so a
    Document that cannot be asked about is never scored as though it had
    produced a poor answer.
    """

    status: int
    category: str
    error: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the refusal payload a client reads."""
        payload: dict[str, Any] = {"error": self.error}
        if self.category:
            payload["category"] = self.category
        return {**payload, **self.detail}


@dataclass(frozen=True)
class AnswerEvent:
    """One thing that happened to a question, named the way the stream names it."""

    name: str
    payload: dict[str, Any]


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
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Bind the answer path to the dependencies the app serves with."""
        self._settings = settings
        self._repositories = repositories
        self._embeddings = embedding_service
        self._vectors = vector_service
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
        """
        refusal = self._refuse_unanswerable(request)
        if refusal is not None:
            return refusal

        file_record = self.files.get_file(request.filename)
        conversation_id = self.conversations.get_conversation_id(file_record["id"])
        if not conversation_id:
            return Refusal(404, "conversation_not_found", "Conversation not found")
        if not isinstance(request.query, str) or not request.query.strip():
            return Refusal(400, "invalid_query", "Query and Filename are required")
        if len(request.query) > MAX_QUERY_CHARS:
            return Refusal(400, "query_too_long", "Query is too long")

        try:
            retrieval, sources, context, prior_turns_text = self._retrieve_context(
                request, file_record, conversation_id
            )
        except (
            VectorStoreUnavailableError,
            VectorStoreConfigurationError,
        ) as exc:
            log.exception("/response vector store failed for %s", request.filename)
            return vector_store_refusal(exc)
        except VectorDimensionError as exc:
            log.warning(
                "/response vector dimension mismatch for %s: %s", request.filename, exc
            )
            return Refusal(409, "vector_dimension_mismatch", str(exc))
        except Exception:
            log.exception("/response retrieval failed for %s", request.filename)
            return Refusal(500, "", "Internal server error")

        if not self._still_present(request.filename):
            log.warning(
                "/response refusing chat for deleting document %s", request.filename
            )
            return deletion_block_refusal(self.files.get_file(request.filename))

        # Decided before the Turn is committed, and acted on below: whether
        # this question gets a model call is settled the moment retrieval is.
        abstention = abstention_for(sources)
        try:
            turn_id = self.conversations.start_turn(conversation_id, request.query)
        except Exception:
            log.exception(
                "/response could not commit the question for %s", request.filename
            )
            return Refusal(500, "", "Internal server error")

        return ResolvedTurn(
            turn_id=turn_id,
            request=request,
            context=context,
            prior_turns=prior_turns_text,
            sources=sources,
            retrieval=retrieval,
            abstention=abstention,
            provenance=self._provenance(request, file_record, retrieval),
        )

    def stream(self, resolved: ResolvedTurn) -> Iterator[AnswerEvent]:
        """
        Yield what happens to one resolved question, ending on one outcome.

        The turn is already on record, so a stream that ends early still leaves
        a Turn the reader can see. A client that walks away mid-answer closes
        the Turn as cancelled rather than leaving it pending forever.
        """
        if resolved.abstention is not None:
            yield from self._abstain(resolved, resolved.abstention)
            return
        yield from self._generate(resolved)

    def _refuse_unanswerable(self, request: AnswerRequest) -> Refusal | None:
        """Return the refusal that stops a question before retrieval."""
        file_record = self.files.get_file(request.filename)
        if not file_record:
            return Refusal(404, "file_not_found", "File not found")
        if is_deleting_record(file_record):
            return deletion_block_refusal(file_record)
        ingestion_job = self._repositories.ingestion_jobs.get_latest(request.filename)
        if (
            file_record.get("is_processed")
            and ingestion_job
            and ingestion_job["state"] in REINDEXING_STATES
        ):
            return Refusal(
                409,
                "document_indexing",
                "This document is being reindexed. Try again when indexing finishes.",
                {"job": ingestion_job},
            )
        # A stale index still holds vectors, but they were built with a parser,
        # model, or collection schema the app no longer serves, so querying it
        # would answer from an incompatible index. A document that was never
        # indexed has nothing to query either.
        state = index_status(self.files, file_record, self._settings)
        if state.is_stale:
            log.info(
                "chat refused for stale index %s: %s", request.filename, state.changes
            )
            return Refusal(
                409,
                "index_stale",
                "This document's index no longer matches the current settings. "
                "Reindex it to ask questions again.",
                {"action": "reindex", "index": state.to_dict(self._settings)},
            )
        if state.state == "pending":
            return Refusal(
                409,
                "index_pending",
                "This document is not indexed yet. Index it to ask questions.",
                {"action": "reindex", "index": state.to_dict(self._settings)},
            )
        return None

    def _retrieve_context(
        self,
        request: AnswerRequest,
        file_record: dict[str, Any],
        conversation_id: str,
    ) -> tuple[Retrieval, list[dict[str, Any]], str, str]:
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
        return retrieval, kept_sources, chat_context.context, chat_context.prior_turns

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
        """Close the Turn as failed and yield the terminal event."""
        self._record_outcome(
            resolved.request.filename,
            self.conversations.fail_turn,
            resolved.turn_id,
            failure.message,
            failure.reason,
        )
        yield AnswerEvent(
            event_name,
            {"error": failure.message, "category": failure.category},
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
        except Exception:
            log.exception(
                "/response could not record the abstention for %s",
                resolved.request.filename,
            )
            yield AnswerEvent(
                "persistence_error",
                {"error": "The answer could not be saved. Please ask again."},
            )
            return
        if not recorded:
            log.warning(
                "/response turn %s already ended for %s",
                resolved.turn_id,
                resolved.request.filename,
            )
            yield AnswerEvent(
                "persistence_error",
                {"error": UNSAVED.message, "category": UNSAVED.category},
            )
            return
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
            return ResolvedAnswer(parsed.answer, citations)
        log.warning(
            "/response citations unresolved for %s: %s",
            resolved.request.filename,
            ",".join(citations.invalid_ids),
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
        try:
            for token in request.provider.stream_response(
                request.query, resolved.context, resolved.prior_turns
            ):
                fragments.append(token)
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
            raise
        except ProviderTimeoutError:
            log.warning("/response generation timed out for %s", request.filename)
            yield from self._fail(resolved, PROVIDER_SLOW, "provider_error")
            return
        except EmptyAnswerError:
            log.warning("/response generation was empty for %s", request.filename)
            yield from self._fail(resolved, PROVIDER_SILENT, "provider_error")
            return
        except Exception as exc:  # noqa: BLE001 - reported as one provider outcome
            log.error(
                "/response generation failed for %s: %s",
                request.filename,
                type(exc).__name__,
            )
            yield from self._fail(resolved, PROVIDER_DOWN, "provider_error")
            return

        answer = "".join(fragments).strip()
        if not self._still_present(request.filename):
            # The Document and its Conversation went away mid-answer, so there
            # is nothing left to store this in. Saying the answer was saved
            # would be a lie the reader would replay from history.
            log.warning(
                "/response abandoning answer for deleted document %s", request.filename
            )
            try:
                self.conversations.cancel_turn(resolved.turn_id, "document deleted")
            except Exception:
                log.exception(
                    "/response could not cancel turn for deleted document %s",
                    request.filename,
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
        except Exception:
            log.exception("failed to persist answer for %s", request.filename)
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
            yield from self._fail(resolved, UNSAVED, "persistence_error")
            return
        yield self._done(resolved, citations, stored)


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
