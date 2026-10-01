"""
The Research Brief: a bounded, tool-using run over two Documents.

A chat answer retrieves once and answers from what it found. A brief hands the
choosing to the model: it searches, reads what it wants to read, searches again,
and answers when it has enough. That is worth having for a question that spans
two papers, and it is exactly the shape that can spend without being asked — so
everything about the reach is decided before the first turn and everything about
the spend is decided while it runs.

The reach is fixed here, not by the model. ``resolve`` takes two Documents and
refuses the brief for anything that would make the pair unreadable; the model is
then shown the pair under two labels and given four tools, all of which can only
read inside it. The spend is bounded by :class:`BriefBudget`, checked before
each turn rather than after it, so a brief that runs out of turns or of time
stops with what it has instead of being cut off mid-answer.

The run has one terminal event, and it is always honest about what happened:
``done`` when the model answered, ``partial`` when a limit stopped it with
evidence already collected, ``cancelled`` when the reader walked away, and
``provider_error`` when the model call failed. A brief that hit a limit is
labelled incomplete rather than presented as a finished answer, because the
difference between the two is the difference between a result a researcher can
rely on and one they will check anyway.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from services.abstention import NO_EVIDENCE
from services.accounts.chat_settings_service import ModelCapabilities
from services.answering import AnswerEvent, Refusal
from services.brief import cancellation
from services.brief.budget import BriefBudget, BriefLimits
from services.brief.compare import QUOTED_CHARS, compare_items
from services.brief.evidence import EvidenceLedger
from services.brief.prompts import (
    BRIEF_PROMPT_VERSION,
    BRIEF_SYSTEM_INSTRUCTION,
    build_brief_prompt,
)
from services.brief.result import brief_from_answer
from services.brief.scope import BriefScope
from services.brief.tools import (
    COMPARE_EVIDENCE,
    READ_PAGE,
    READ_PASSAGES,
    REFUSED_INVALID_ARGUMENTS,
    REFUSED_OUT_OF_SCOPE,
    REFUSED_REPEATED,
    REFUSED_UNKNOWN_EVIDENCE,
    REFUSED_UNKNOWN_PAGE,
    REFUSED_CANCELLED,
    REFUSED_NOT_RUN,
    REFUSED_UNKNOWN_TOOL,
    REFUSAL_MESSAGES,
    SEARCH_PASSAGES,
    TOOL_SPECS,
    ToolRefusal,
    call_fingerprint,
    refuse,
    render_result,
    validate_arguments,
)
from services.llm.base import LLMProvider, ProviderTimeoutError
from services.llm.tools import (
    ToolMessage,
    ToolTurn,
    assistant_message,
    render_tool_result,
    user_message,
)
from services.retrieval.base import RetrievalResult
from services.telemetry.brief import (
    CANCELLED,
    COMPLETE,
    FAILED,
    INCOMPLETE,
    BriefTrace,
)
from services.telemetry.factory import tracer_for
from services.telemetry.spans import error_category

log = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 8192

#: How much of one Page a read_page call shows. Enough to carry the Page's
#: argument, bounded so a long Page cannot cost the brief a whole Document.
PAGE_TEXT_CHARS = 4000

#: How much fallback context a failed one-shot call is allowed to quote.
_FALLBACK_TOKENS_PER_CHAR = 0.25

#: The monotonic clock every brief measures its wall clock against. It is a
#: module-level name rather than a private import so a test can replace it and
#: run a brief out of its time budget without waiting for one to elapse, which
#: is the only way to assert the ceiling is enforced between turns.
CLOCK: Callable[[], float] = time.monotonic


@dataclass(frozen=True)
class BriefRequest:
    """One cross-Document question, with the pair of Documents and the model."""

    filenames: tuple[str, ...]
    question: str
    provider: LLMProvider
    model: ModelCapabilities
    limits: BriefLimits = field(default_factory=BriefLimits)


@dataclass(frozen=True)
class ResolvedBrief:
    """A brief that reached the model, with the scope and clock it runs under."""

    request: BriefRequest
    scope: BriefScope
    ledger: EvidenceLedger
    trace: BriefTrace
    #: Set when the reader asks this brief to stop. Checked between turns and
    #: between tool calls, the two points where stopping leaves the evidence
    #: consistent.
    cancel: threading.Event


@dataclass(frozen=True)
class PageTarget:
    """One Page read the brief may make: the Document, the Page, its pages."""

    document: Any
    page: int
    pages: tuple[str, ...]


class BriefService:
    """
    Runs one bounded Research Brief over two Documents.

    The service owns the scope, the tool dispatch, the budget, and the terminal
    outcome. The route supplies the provider and the HTTP shape; the trace is
    injected so a test reads what the application would have exported.
    """

    def __init__(
        self,
        *,
        settings: Any,
        repositories: Any,
        embedding_service: Any,
        vector_service: Any,
        tracer: Any = None,
        clock: Callable[[], float] | None = None,
        storage: Any = None,
        parser: Any = None,
    ) -> None:
        """
        Bind the brief to the dependencies the app serves with.

        ``tracer`` is where each brief's trace goes. Left out, the brief builds
        the one the running configuration asks for, so a deployment configures
        observability in settings rather than in each caller. ``clock`` is the
        brief's monotonic reading; left out, every brief reads the same clock.
        ``storage`` and ``parser`` are what a Page read parses; left out, a
        Page read fails rather than reading from anywhere else, because there
        is nowhere else a brief is allowed to read from.
        """
        self._settings = settings
        self._repositories = repositories
        self._embeddings = embedding_service
        self._vectors = vector_service
        self._tracer = tracer if tracer is not None else tracer_for(settings)
        self._clock_override = clock
        self._storage = storage
        self._parser = parser

    def _now(self) -> float:
        """
        Return this brief's monotonic reading.

        The module clock is read at call time rather than bound at construction,
        so a service composed at boot still reads whatever clock is installed
        when the brief runs.
        """
        return self._clock_override() if self._clock_override is not None else CLOCK()

    @property
    def files(self) -> Any:
        """Return the Document store the scope is judged from."""
        return self._repositories.files

    def resolve(
        self, request: BriefRequest, scope: BriefScope
    ) -> ResolvedBrief | Refusal:
        """
        Decide everything a brief needs before its first model turn.

        Two refusals happen here rather than in the loop, because both are
        properties of the request rather than of a turn: a model the catalog
        says cannot run a brief, and a question the app will not put to a model
        at all. Everything else — whether the pair is readable, how much the run
        may spend — is either already decided by the caller or decided by the
        budget as it goes.
        """
        trace = BriefTrace.start(self._tracer, model=request.model)
        # The question is recorded so the trace shows there was one; the
        # redactor replaces it with a fingerprint unless a local capture window
        # is open.
        trace.identify(question=request.question)
        blockers = request.model.research_brief_blockers()
        if blockers:
            return self._refused(
                trace,
                Refusal(
                    409,
                    "research_brief_unsupported_model",
                    (
                        "A research brief needs a model that can call tools and "
                        "write structured output. Choose one in Settings that "
                        "reports both."
                    ),
                    {"missing_capabilities": list(blockers)},
                ),
            )
        question = (request.question or "").strip()
        if not question:
            return self._refused(
                trace,
                Refusal(400, "invalid_question", "A research question is required"),
            )
        if len(question) > MAX_QUESTION_CHARS:
            return self._refused(
                trace,
                Refusal(400, "question_too_long", "The research question is too long"),
            )
        trace.identify(documents=[item.to_dict() for item in scope.documents])
        trace.scope(scope.to_dict())
        return ResolvedBrief(
            request=request,
            scope=scope,
            ledger=EvidenceLedger(),
            trace=trace,
            cancel=cancellation.register(trace.trace_id),
        )

    @staticmethod
    def _refused(trace: BriefTrace, refusal: Refusal) -> Refusal:
        """Record a refused brief and return the refusal unchanged."""
        trace.refused(refusal.category, refusal.status)
        return refusal

    def stream(self, resolved: ResolvedBrief) -> Iterator[AnswerEvent]:
        """
        Yield what happens to one brief, ending on one outcome.

        The budget is charged before each turn rather than after it, so a brief
        that has spent its allowance stops instead of starting a turn it cannot
        finish. A reader who walks away closes the run as cancelled and keeps
        whatever was on screen, marked as not finished.
        """
        request = resolved.request
        resolved.trace.identify(question=request.question)
        resolved.trace.begin(self._now)
        budget = BriefBudget(limits=request.limits, clock=self._now)
        cancel = resolved.cancel
        conversation: list[ToolMessage] = [
            user_message(
                build_brief_prompt(request.question, resolved.scope.describe())
            )
        ]
        stopped_by = ""
        try:
            # Opened inside the guarded block: a reader who walks away between
            # this event and the first turn is still a cancelled run, and the
            # trace has to say so.
            yield AnswerEvent(
                "start",
                {
                    "brief_id": resolved.trace.trace_id,
                    "documents": resolved.scope.to_dict(),
                    "prompt_version": BRIEF_PROMPT_VERSION,
                    "limits": budget.limits.to_dict(),
                },
            )
            while True:
                reason = self._stop_reason(budget, cancel)
                if reason is not None:
                    stopped_by = reason
                    break
                turn = budget.turns + 1
                budget.charge_turn()
                try:
                    answer = self._turn(resolved, budget, conversation, turn, cancel)
                except (ProviderTimeoutError, TimeoutError) as exc:
                    yield from self._provider_failure(resolved, budget, exc, turn)
                    return
                except Exception as exc:  # noqa: BLE001 - one provider outcome
                    log.error(
                        "/research turn failed for %s: %s",
                        ",".join(request.filenames),
                        type(exc).__name__,
                    )
                    yield from self._provider_failure(resolved, budget, exc, turn)
                    return
                if answer is not None:
                    if cancel.is_set():
                        stopped_by = "cancelled"
                        break
                    yield from self._complete(resolved, budget, answer, turn)
                    return
            yield from self._stopped(resolved, budget, stopped_by)
        except GeneratorExit:
            # The reader left mid-brief. Whatever was on screen stays there and
            # is marked incomplete by the client, and the trace says the run was
            # cancelled rather than leaving it looking like a finished answer.
            resolved.trace.budget(
                budget, stopped="cancelled", evidence=resolved.ledger.held()
            )
            resolved.trace.outcome(
                CANCELLED, reason="client disconnected", turns=budget.turns
            )
            raise
        finally:
            cancellation.release(resolved.trace.trace_id)
            resolved.trace.finish()

    @staticmethod
    def _stop_reason(budget: BriefBudget, cancel: threading.Event) -> str | None:
        """
        Return why the loop must stop, the reader's cancellation included.

        A cancel is checked before the budget on purpose: the reader pressing
        stop is a decision, and it should not have to wait for whichever limit
        happened to be closest. The budget is still charged and recorded, so a
        cancelled brief says what it had spent.
        """
        if cancel.is_set():
            return "cancelled"
        return budget.stop_reason()

    def _turn(
        self,
        resolved: ResolvedBrief,
        budget: BriefBudget,
        conversation: list[ToolMessage],
        turn: int,
        cancel: threading.Event,
    ) -> str | None:
        """
        Run one model turn and any tool calls it asked for.

        Returns the model's answer when it produced one, and None when it asked
        for tools instead — the loop keeps going in that case, which is the
        whole point of a brief. The conversation is appended to in place so the
        next turn sees what this one searched for and what came back.
        """
        request = resolved.request
        model_turn = request.provider.complete_with_tools(
            conversation,
            TOOL_SPECS,
            system_instruction=BRIEF_SYSTEM_INSTRUCTION,
        )
        budget.charge_tokens(self._turn_tokens(conversation, model_turn))
        conversation.append(assistant_message(model_turn.text, model_turn.tool_calls))
        if not model_turn.tool_calls:
            return model_turn.text.strip()
        answered: set[str] = set()
        for call in model_turn.tool_calls:
            conversation.append(
                render_tool_result(
                    call, self._run_tool(resolved, budget, call, turn, cancel)
                )
            )
            answered.add(call.id)
            # Checked between calls as well as between turns: one turn can carry
            # several calls, and a model that asks for ten at once would spend
            # ten retrievals against a budget meant to bound them.
            if budget.tool_calls >= budget.limits.max_tool_calls or cancel.is_set():
                break
        for call in model_turn.tool_calls:
            if call.id in answered:
                continue
            # Every call the model made is answered, including the ones a limit
            # stopped. Both providers pair a result to a call by id and reject a
            # turn whose calls went unanswered, so leaving one out would fail
            # the next request rather than ending the brief cleanly. Each is
            # recorded too: what the model reached for is on the trace whether or
            # not it ran.
            with resolved.trace.tool(
                turn=turn,
                name=call.name,
                arguments=call.arguments,
                fingerprint=call_fingerprint(call.name, call.arguments),
            ) as span:
                resolved.trace.refused_tool(span, REFUSED_NOT_RUN)
            conversation.append(
                render_tool_result(
                    call,
                    render_result({"refused": REFUSAL_MESSAGES[REFUSED_NOT_RUN]}),
                )
            )
        return None

    def _turn_tokens(
        self, conversation: list[ToolMessage], model_turn: ToolTurn
    ) -> int:
        """
        Return what one turn read and wrote, from the provider or from a count.

        Both providers report usage, so the reported numbers are what a brief is
        normally billed on. A provider that reported none is counted from the
        text instead, which is the same estimate the answer path uses, so the
        two paths price a turn the same way rather than one being free.
        """
        if model_turn.input_tokens or model_turn.output_tokens:
            return model_turn.input_tokens + model_turn.output_tokens
        characters = len(model_turn.text) + sum(
            len(message.text) for message in conversation
        )
        return int(characters * _FALLBACK_TOKENS_PER_CHAR)

    def _run_tool(
        self,
        resolved: ResolvedBrief,
        budget: BriefBudget,
        call: Any,
        turn: int,
        cancel: threading.Event,
    ) -> str:
        """
        Run one tool call, or refuse it, and return what the model reads.

        Every refusal is a value the model receives rather than an exception the
        loop raises: a model that asked for something out of scope needs to be
        told what is in scope, not failed. The refusal is recorded on the tool
        span, so what the model reached for is on the trace whether or not it
        was allowed.
        """
        fingerprint = call_fingerprint(call.name, call.arguments)
        with resolved.trace.tool(
            turn=turn, name=call.name, arguments=call.arguments, fingerprint=fingerprint
        ) as span:
            # Checked before anything else: a call the reader has already
            # cancelled is not dispatched, and saying why costs the model one
            # turn rather than leaving it to guess.
            if cancel.is_set():
                resolved.trace.refused_tool(span, REFUSED_CANCELLED)
                return render_result({"refused": REFUSAL_MESSAGES[REFUSED_CANCELLED]})
            refusal = self._check(resolved, budget, call, fingerprint)
            if refusal is not None:
                resolved.trace.refused_tool(span, refusal.reason)
                return refusal.to_result()
            target: PageTarget | None = None
            if call.name == READ_PAGE:
                # Resolved before the charge, so a refused Page costs a turn
                # but not one of the brief's tool calls: it read nothing.
                target, refusal = self._page_target(resolved, call)
                if refusal is not None:
                    resolved.trace.refused_tool(span, refusal.reason)
                    return refusal.to_result()
                assert target is not None
            # Charged after the checks, so a refused call costs a turn but not
            # one of the brief's tool calls: it searched nothing.
            budget.charge_tool_call(fingerprint)
            if call.name == SEARCH_PASSAGES:
                payload = self._search(resolved, call, span)
            elif call.name == READ_PAGE:
                assert target is not None
                payload = self._read_page(resolved, call, target, span)
            elif call.name == COMPARE_EVIDENCE:
                payload = self._compare(resolved, call, span)
            else:
                payload = self._read(resolved, call, span)
            return render_result(payload)

    def _check(
        self,
        resolved: ResolvedBrief,
        budget: BriefBudget,
        call: Any,
        fingerprint: str,
    ) -> ToolRefusal | None:
        """
        Return why this call would not be made, or None when it will be.

        The order is deliberate: what the call names is checked before what it
        costs. An unknown tool is refused whatever its arguments, and a repeated
        call is refused before a search spends a retrieval against a budget that
        has already been spent on it.
        """
        if call.name not in (
            SEARCH_PASSAGES,
            READ_PASSAGES,
            READ_PAGE,
            COMPARE_EVIDENCE,
        ):
            return refuse(REFUSED_UNKNOWN_TOOL)
        spec = next(item for item in TOOL_SPECS if item.name == call.name)
        invalid = validate_arguments(spec, call.arguments)
        if invalid is not None:
            return invalid
        if budget.repeats_call(fingerprint) >= budget.limits.max_repeated_calls:
            return refuse(REFUSED_REPEATED)
        if call.name in (SEARCH_PASSAGES, READ_PAGE):
            if resolved.scope.document_for_label(call.arguments["label"]) is None:
                return refuse(REFUSED_OUT_OF_SCOPE)
        else:
            evidence_ids = call.arguments["evidence_ids"]
            if call.name == COMPARE_EVIDENCE and len(set(evidence_ids)) != len(
                evidence_ids
            ):
                # Comparing evidence with itself is not a comparison: the
                # schema asks for distinct ids, and a repeated one is refused
                # rather than arranged.
                return refuse(REFUSED_INVALID_ARGUMENTS)
            for evidence_id in evidence_ids:
                if resolved.ledger.get(evidence_id) is None:
                    return refuse(REFUSED_UNKNOWN_EVIDENCE)
        return None

    def _search(self, resolved: ResolvedBrief, call: Any, span: Any) -> dict[str, Any]:
        """
        Search one scoped Document and admit what came back as evidence.

        The Document is chosen by the label the model named and looked up in the
        scope, so a search can only ever run against one of the two. Retrieval
        failures propagate: a brief whose search cannot run has no evidence and
        the loop turns that into a terminal event, rather than reporting a
        search that found nothing because the store was down.
        """
        document = resolved.scope.document_for_label(call.arguments["label"])
        assert document is not None  # checked in _check
        query = call.arguments["query"]
        file_record = self.files.get_file(document.filename) or {}
        embedding = self._embeddings.embed_texts(query)[0]
        started = resolved.trace.clock()
        result: RetrievalResult = self._vectors.query_vectors(
            embedding,
            document.filename,
            query_text=query,
            generation=file_record.get("index_generation"),
            include_legacy=file_record.get("index_generation") is None,
        )
        latency_ms = round((resolved.trace.clock() - started) * 1000, 3)
        evidence = resolved.ledger.admit(
            result.sources, document=document, method=result.method
        )
        resolved.trace.searched(
            span,
            label=document.label,
            document_id=document.document_id,
            method=result.method,
            outcome=result.outcome,
            candidate_count=len(result.candidates),
            latency_ms=latency_ms,
            evidence=evidence,
        )
        # The excerpt is what search returns; `read_passages` is how the model
        # gets the rest. Returning the whole Passage from a search would make
        # the second tool pointless and a search unbounded in what it costs.
        return {
            "results": [
                {
                    "evidence_id": item.evidence_id,
                    "label": item.label,
                    "rank": item.rank,
                    "page": item.page,
                    "excerpt": item.excerpt,
                }
                for item in evidence
            ],
            "retrieval_method": result.method,
            "note": (
                "These are the first part of each Passage. Call read_passages "
                "with the ids you need in full before relying on one."
            ),
        }

    def _read(self, resolved: ResolvedBrief, call: Any, span: Any) -> dict[str, Any]:
        """Return the full text of Passages this brief already holds."""
        held = [
            evidence
            for evidence_id in call.arguments["evidence_ids"]
            if (evidence := resolved.ledger.read(evidence_id)) is not None
        ]
        resolved.trace.read(span, evidence_ids=[item.evidence_id for item in held])
        return {
            "passages": [
                {
                    "evidence_id": item.evidence_id,
                    "label": item.label,
                    "page": item.page,
                    "passage": item.shown,
                }
                for item in held
            ]
        }

    def _page_target(
        self, resolved: ResolvedBrief, call: Any
    ) -> tuple[PageTarget | None, ToolRefusal | None]:
        """
        Return the Page one read_page call may read, or why it may not.

        The label was already checked against the scope; what is checked here
        needs the Document itself. The file must still exist, its active index
        generation must still be the one the scope was resolved with — a
        Document reindexed since the brief started is no longer the Document
        the model was shown — and the Page must lie within the Document's own
        bounds. A Page that fails any of these is refused before anything is
        charged for reading it.
        """
        document = resolved.scope.document_for_label(call.arguments["label"])
        if document is None:  # checked in _check
            return None, refuse(REFUSED_OUT_OF_SCOPE)
        file_record = self.files.get_file(document.filename) or {}
        if not file_record:
            return None, refuse(REFUSED_OUT_OF_SCOPE)
        # The active generation must still be the one the scope was resolved
        # with: a Document reindexed since the brief started is no longer the
        # Document the model was shown, whatever its generation now reads.
        if file_record.get("index_generation") != document.index_generation:
            return None, refuse(REFUSED_OUT_OF_SCOPE)
        if self._storage is None or self._parser is None:
            raise RuntimeError("page reading is not configured for this brief")
        try:
            file_bytes = self._storage.open(document.filename)
            pages = tuple(
                self._parser.resolve(document.filename, file_bytes).extract_pages(
                    file_bytes
                )
            )
        except Exception:
            return None, refuse(REFUSED_OUT_OF_SCOPE)
        page = call.arguments["page"]
        if not isinstance(page, int) or isinstance(page, bool):
            return None, refuse(REFUSED_UNKNOWN_PAGE)
        if page < 1 or page > len(pages):
            return None, refuse(REFUSED_UNKNOWN_PAGE)
        return PageTarget(document=document, page=page, pages=pages), None

    def _read_page(
        self,
        resolved: ResolvedBrief,
        call: Any,
        target: PageTarget,
        span: Any,
    ) -> dict[str, Any]:
        """
        Read one Page of a scoped Document and admit it as evidence.

        The Page text is bounded before it is admitted, so a long Page cannot
        cost the brief a whole Document. The admitted evidence carries the
        Document, the Page, and its provenance, which is what lets the final
        brief cite Page text with the same ids it cites Passages with.
        """
        started = resolved.trace.clock()
        raw = target.pages[target.page - 1] or ""
        total_chars = len(raw)
        text = raw[:PAGE_TEXT_CHARS]
        truncated = total_chars > PAGE_TEXT_CHARS
        latency_ms = round((resolved.trace.clock() - started) * 1000, 3)
        if not text.strip():
            resolved.trace.read(span, evidence_ids=[])
            return {
                "page": {
                    "evidence_id": None,
                    "label": target.document.label,
                    "page": target.page,
                    "page_count": len(target.pages),
                    "text": "",
                    "truncated": False,
                    "total_chars": total_chars,
                },
                "note": "That Page has no text.",
            }
        evidence = resolved.ledger.admit_page(
            document=target.document, page=target.page, content=text
        )
        resolved.trace.paged(
            span,
            evidence_id=evidence.evidence_id,
            label=target.document.label,
            document_id=target.document.document_id,
            page=target.page,
            page_count=len(target.pages),
            latency_ms=latency_ms,
            truncated=truncated,
        )
        return {
            "page": {
                "evidence_id": evidence.evidence_id,
                "label": evidence.label,
                "page": evidence.page,
                "page_count": len(target.pages),
                "text": evidence.content,
                "truncated": truncated,
                "total_chars": total_chars,
            },
            "note": (
                "Cite this Page with its evidence id. "
                + (
                    "The Page was truncated to its first part."
                    if truncated
                    else "This is the Page in full."
                )
            ),
        }

    def _compare(self, resolved: ResolvedBrief, call: Any, span: Any) -> dict[str, Any]:
        """
        Arrange held evidence side by side, collecting nothing new.

        Every id was already checked against the ledger, so everything named
        here is evidence the brief holds. Fully quoted evidence is marked
        read, because the comparison shows its text in full; evidence longer
        than the quote keeps whatever it had been shown. The ledger itself is
        unchanged otherwise: a comparison links evidence, it never admits it.
        """
        held = [
            evidence
            for evidence_id in call.arguments["evidence_ids"]
            if (evidence := resolved.ledger.get(evidence_id)) is not None
        ]
        started = resolved.trace.clock()
        comparison = compare_items(held)
        latency_ms = round((resolved.trace.clock() - started) * 1000, 3)
        for item in held:
            if len(item.content) <= QUOTED_CHARS:
                resolved.ledger.read(item.evidence_id)
        resolved.trace.compared(
            span,
            evidence_ids=[item.evidence_id for item in held],
            latency_ms=latency_ms,
        )
        return comparison

    def _model_metadata(self, resolved: ResolvedBrief) -> dict[str, Any]:
        """Return the model that ran this brief, as the client persists it."""
        model = resolved.request.model
        return {
            "provider": getattr(model, "provider", ""),
            "model": getattr(model, "id", ""),
        }

    def _complete(
        self, resolved: ResolvedBrief, budget: BriefBudget, answer: str, turn: int
    ) -> Iterator[AnswerEvent]:
        """
        Close a brief the model answered, or report that it had no evidence.

        A brief whose tools collected nothing has nothing to cite, so it says so
        in the app's own words rather than showing prose that looks grounded. That
        is the same distinction the chat path draws, and it costs no model call:
        the answer is discarded rather than stored, because storing it would put
        an uncitable result in the transcript as though it were evidence-backed.

        A brief that did collect evidence is parsed into its structured shape
        before it is sent: the summary stays readable as the answer, and the
        claims, gaps, and abstention status travel beside it so each part can be
        checked against the evidence ids this run collected.
        """
        if not answer:
            yield from self._provider_failure(
                resolved,
                budget,
                RuntimeError("the model returned no text"),
                turn,
            )
            return
        if not resolved.ledger.ids():
            resolved.trace.budget(budget, evidence=resolved.ledger.held())
            resolved.trace.outcome(INCOMPLETE, reason="no evidence", turns=turn)
            yield AnswerEvent(
                "abstained",
                {
                    "abstained": True,
                    "message": (
                        "Neither selected Document contains evidence that answers "
                        "this question, so there is nothing to write from."
                    ),
                    "reason": NO_EVIDENCE,
                    "documents": resolved.scope.to_dict(),
                    "prompt_version": BRIEF_PROMPT_VERSION,
                    "model": self._model_metadata(resolved),
                },
            )
            return
        structured, invalid_ids = brief_from_answer(answer, resolved.ledger.ids())
        resolved.trace.budget(budget, evidence=resolved.ledger.held())
        resolved.trace.outcome(COMPLETE, turns=turn)
        yield AnswerEvent(
            "done",
            {
                "done": True,
                "status": COMPLETE,
                "answer": structured.summary or answer.strip(),
                "brief": structured.to_dict(),
                "claims": [claim.to_dict() for claim in structured.claims],
                "gaps": list(structured.gaps),
                "abstained": structured.abstained,
                "invalid_citations": list(invalid_ids),
                "evidence": resolved.ledger.to_dict(),
                "documents": resolved.scope.to_dict(),
                "prompt_version": BRIEF_PROMPT_VERSION,
                "model": self._model_metadata(resolved),
                "budget": budget.usage(),
            },
        )

    def _stopped(
        self, resolved: ResolvedBrief, budget: BriefBudget, stopped_by: str
    ) -> Iterator[AnswerEvent]:
        """
        Close a brief a limit stopped, keeping what it collected.

        The status is ``incomplete`` and the reason is the limit that ended it,
        so a partial brief is never presented as a finished one. When nothing was
        collected there is no partial result to show and the brief says the
        evidence was never found.
        """
        resolved.trace.budget(
            budget, stopped=stopped_by, evidence=resolved.ledger.held()
        )
        resolved.trace.outcome(INCOMPLETE, reason=stopped_by, turns=budget.turns)
        if not resolved.ledger.ids():
            yield AnswerEvent(
                "abstained",
                {
                    "abstained": True,
                    "message": (
                        f"Neither selected Document returned evidence before the "
                        f"brief ran out ({stopped_by.replace('_', ' ')})."
                    ),
                    "reason": NO_EVIDENCE,
                    "documents": resolved.scope.to_dict(),
                    "stopped_by": stopped_by,
                    "prompt_version": BRIEF_PROMPT_VERSION,
                    "model": self._model_metadata(resolved),
                },
            )
            return
        yield AnswerEvent(
            "done",
            {
                "done": True,
                "status": INCOMPLETE,
                "complete": False,
                "stopped_by": stopped_by,
                "message": (
                    f"This brief was stopped before it finished "
                    f"({stopped_by.replace('_', ' ')}). The evidence below is "
                    "what it collected."
                ),
                "brief": {
                    "summary": "",
                    "claims": [],
                    "gaps": [],
                    "abstained": False,
                },
                "claims": [],
                "gaps": [],
                "abstained": False,
                "evidence": resolved.ledger.to_dict(),
                "documents": resolved.scope.to_dict(),
                "prompt_version": BRIEF_PROMPT_VERSION,
                "model": self._model_metadata(resolved),
                "budget": budget.usage(),
            },
        )

    def _provider_failure(
        self,
        resolved: ResolvedBrief,
        budget: BriefBudget,
        error: BaseException,
        turn: int,
    ) -> Iterator[AnswerEvent]:
        """
        Close a brief whose model call failed, naming the kind of failure.

        Whatever the brief collected before the failure is still sent, marked
        incomplete: a reader who has seen evidence arrive and then hit a
        provider error should be able to look at it rather than be shown an
        empty result.
        """
        category = error_category(error)
        resolved.trace.budget(budget, stopped=category, evidence=resolved.ledger.held())
        resolved.trace.outcome(FAILED, reason=category, turns=turn)
        yield AnswerEvent(
            "provider_error",
            {
                "error": (
                    "The language model could not continue this brief. Try again."
                ),
                "category": ("timeout" if category == "timeout" else "provider"),
                "status": INCOMPLETE,
                "brief": {
                    "summary": "",
                    "claims": [],
                    "gaps": [],
                    "abstained": False,
                },
                "claims": [],
                "gaps": [],
                "abstained": False,
                "evidence": resolved.ledger.to_dict(),
                "documents": resolved.scope.to_dict(),
                "prompt_version": BRIEF_PROMPT_VERSION,
                "model": self._model_metadata(resolved),
                "budget": budget.usage(),
            },
        )
