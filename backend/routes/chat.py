"""HTTP routes for document chat and message history."""

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from flask import Flask, Response, jsonify, request, stream_with_context

from routes.common import (
    deletion_blocked_response,
    is_deleting_record,
    traversal_check,
    vector_store_error_response,
)
from services.accounts.chat_settings_service import (
    SUPPORTED_MODELS,
    SettingsError,
    model_for,
)
from services.accounts.secrets_service import (
    SecretsResaveRequiredError,
    decrypt_api_key,
)
from services.indexing.state import index_status
from services.abstention import abstention_for
from services.citations import (
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
from services.llm.base import ChatCredentials, EmptyAnswerError, ProviderTimeoutError
from services.chat_context import build_chat_context, build_model_rewriter
from services.retrieval.base import (
    VectorDimensionError,
    VectorStoreConfigurationError,
    VectorStoreUnavailableError,
)
from services.retrieval.query_expansion import expand_query

if TYPE_CHECKING:
    from composition import Services

log = logging.getLogger(__name__)

MAX_QUERY_CHARS = 8192


@dataclass(frozen=True)
class AnswerFailure:
    """What one way an answer can fail shows the user and records on the Turn."""

    message: str
    reason: str


@dataclass(frozen=True)
class ResolvedAnswer:
    """One generated answer: the prose to show, and the claims that survived."""

    answer: str
    citations: ValidatedCitations


def _normalize_source(source: dict) -> dict:
    if "page" in source:
        page = source["page"]
    else:
        page = source.get("page_no")
    return {**source, "page": page if page is not None else None}


def register_chat_routes(app: Flask, services: "Services") -> None:
    """Register chat response and message-history routes."""
    app_settings_repository = services.repositories.app_settings
    files_repository = services.repositories.files
    conversations_repository = services.repositories.conversations
    ingestion_jobs = services.repositories.ingestion_jobs
    embedding_service = services.embedding_service
    vector_service = services.vector_service
    chat_provider_factory = services.chat_provider_factory

    @app.route("/response", methods=["POST"])
    def get_response():
        data = request.get_json() or {}
        query = data.get("query")
        filename = data.get("filename")
        if not query or not filename:
            return jsonify({"error": "Query and Filename are required"}), 400

        guard = traversal_check(services.storage, filename)
        if guard is not None:
            return guard

        try:
            stored = app_settings_repository.get_app_settings()
        except Exception as exc:
            log.error("/response settings read failed: %s", type(exc).__name__)
            return jsonify({"error": "Stored settings could not be read."}), 500
        if not stored:
            return jsonify(
                {
                    "error": (
                        "No chat provider configured. Add a provider and API key "
                        "in Settings."
                    )
                }
            ), 400
        try:
            provider = stored["provider"]
            model = stored["model"]
            ciphertext = stored["encrypted_api_key"]
        except (AttributeError, KeyError, TypeError):
            return jsonify(
                {
                    "error": (
                        "Saved provider settings are incomplete. Re-save your "
                        "provider settings in Settings."
                    )
                }
            ), 400
        if (
            not isinstance(provider, str)
            or not provider
            or not isinstance(model, str)
            or not model
            or not isinstance(ciphertext, str)
            or not ciphertext
        ):
            return jsonify(
                {
                    "error": (
                        "Saved provider settings are incomplete. Re-save your "
                        "provider settings in Settings."
                    )
                }
            ), 400
        try:
            api_key = decrypt_api_key(ciphertext)
        except SecretsResaveRequiredError as exc:
            log.warning("/response stale key derivation")
            return jsonify({"error": str(exc), "needs_resave": True}), 400
        except Exception as exc:
            log.error("/response decrypt failed: %s", type(exc).__name__)
            return jsonify(
                {
                    "error": (
                        "Stored API key could not be decrypted. Re-save your "
                        "provider settings, then try again."
                    )
                }
            ), 500
        if provider not in SUPPORTED_MODELS:
            return jsonify(
                {
                    "error": (
                        "Saved provider settings use an unsupported provider. "
                        "Choose a current provider in Settings."
                    )
                }
            ), 400
        try:
            model_definition = model_for(provider, model)
        except SettingsError:
            return jsonify(
                {
                    "error": (
                        "Saved provider settings use an unsupported model. "
                        "Choose a current model in Settings."
                    )
                }
            ), 400
        try:
            credentials = ChatCredentials(
                provider=provider,
                model=model,
                api_key=api_key,
                verification_timeout_seconds=model_definition.timeout_seconds,
                budget=model_definition.chat_budget(),
            )
            chat_provider = chat_provider_factory(credentials)
        except ValueError as exc:
            log.warning(
                "/response provider error for %s: %s",
                filename,
                type(exc).__name__,
            )
            return jsonify(
                {
                    "error": (
                        "Saved provider settings use an unsupported provider. "
                        "Choose a current provider in Settings."
                    )
                }
            ), 400
        except Exception as exc:
            log.error(
                "/response provider build failed for %s: %s",
                filename,
                type(exc).__name__,
            )
            return jsonify({"error": "Internal server error"}), 500

        file_record = files_repository.get_file(filename)
        if not file_record:
            return jsonify({"error": "File not found"}), 404
        if is_deleting_record(file_record):
            return deletion_blocked_response(file_record)
        ingestion_job = ingestion_jobs.get_latest(filename)
        if (
            file_record.get("is_processed")
            and ingestion_job
            and ingestion_job["state"] in ("queued", "running", "cancelling")
        ):
            return jsonify(
                {
                    "error": "This document is being reindexed. Try again when indexing finishes.",
                    "category": "document_indexing",
                    "job": ingestion_job,
                }
            ), 409
        # A stale index still holds vectors, but they were built with a parser,
        # model, or collection schema the app no longer serves, so querying it
        # would answer from an incompatible index. A document that was never
        # indexed has nothing to query either.
        state = index_status(files_repository, file_record, services.settings)
        if state.is_stale:
            log.info("chat refused for stale index %s: %s", filename, state.changes)
            return jsonify(
                {
                    "error": (
                        "This document's index no longer matches the current "
                        "settings. Reindex it to ask questions again."
                    ),
                    "category": "index_stale",
                    "action": "reindex",
                    "index": state.to_dict(services.settings),
                }
            ), 409
        if state.state == "pending":
            return jsonify(
                {
                    "error": (
                        "This document is not indexed yet. Index it to ask questions."
                    ),
                    "category": "index_pending",
                    "action": "reindex",
                    "index": state.to_dict(services.settings),
                }
            ), 409
        conversation_id = conversations_repository.get_conversation_id(
            file_record["id"]
        )
        if not conversation_id:
            return jsonify({"error": "Conversation not found"}), 404
        if not isinstance(query, str) or not query.strip():
            return jsonify({"error": "Query and Filename are required"}), 400
        if len(query) > MAX_QUERY_CHARS:
            return jsonify({"error": "Query is too long"}), 400

        limits = services.settings.query_context
        try:
            recent_turns = conversations_repository.get_recent_turns(
                conversation_id, limits.recent_turns
            )
            turns_in_conversation = conversations_repository.count_answered_turns(
                conversation_id
            )
        except Exception:
            log.exception("/response conversation read failed for %s", filename)
            return jsonify({"error": "Internal server error"}), 500
        # Two different windows, deliberately: expansion takes the recent
        # questions up to the Turn limit, capped by max_expansion_chars; the
        # transcript sent to the model is bounded by prior_turns_token_budget.
        prior_questions = [
            turn["question"] for turn in recent_turns if turn.get("question")
        ]
        rewriter = (
            build_model_rewriter(chat_provider)
            if limits.query_rewrite and prior_questions
            else None
        )
        expansion = expand_query(
            query,
            prior_questions,
            max_chars=limits.max_expansion_chars,
            rewrite=rewriter,
        )

        try:
            query_embedding = embedding_service.embed_texts(expansion.expanded_query)[0]
            retrieval_result = vector_service.query_vectors(
                query_embedding,
                filename,
                query_text=expansion.expanded_query,
                generation=file_record.get("index_generation"),
                include_legacy=file_record.get("index_generation") is None,
            )
            sources = assign_source_ids(
                [_normalize_source(source) for source in retrieval_result.sources]
            )
            chat_context = build_chat_context(
                query,
                sources,
                recent_turns,
                expansion,
                max_turns=limits.recent_turns,
                prior_turns_token_budget=limits.prior_turns_token_budget,
                context_token_budget=limits.context_token_budget,
                turns_in_conversation=turns_in_conversation,
                input_token_budget=model_definition.max_input_tokens,
            )
            # Citations must match what the model actually saw, so a Citation
            # Source the budget dropped is not stored against the answer.
            sources = list(chat_context.sources)
            context = chat_context.context
            prior_turns_text = chat_context.prior_turns
            retrieval = {
                "method": retrieval_result.method,
                "outcome": retrieval_result.outcome,
                **expansion.to_dict(),
                "dropped_turns": chat_context.dropped_turns,
                "dropped_sources": chat_context.dropped_sources,
            }
        except (
            VectorStoreUnavailableError,
            VectorStoreConfigurationError,
        ) as exc:
            log.exception("/response vector store failed for %s", filename)
            return vector_store_error_response(exc)
        except VectorDimensionError as exc:
            log.warning("/response vector dimension mismatch for %s: %s", filename, exc)
            return jsonify(
                {"error": str(exc), "category": "vector_dimension_mismatch"}
            ), 409
        except Exception:
            log.exception("/response retrieval failed for %s", filename)
            return jsonify({"error": "Internal server error"}), 500

        def _still_present() -> bool:
            """Return whether the document survived a concurrent deletion."""
            try:
                current = files_repository.get_file(filename)
            except Exception:
                log.exception("/response deletion race check failed for %s", filename)
                return True
            if current is None:
                return False
            return not is_deleting_record(current)

        if not _still_present():
            log.warning("/response refusing chat for deleting document %s", filename)
            return deletion_blocked_response(
                files_repository.get_file(filename) or {"deletion_state": "deleting"}
            )
        # Decided before the Turn is committed, and acted on below: whether
        # this question gets a model call is settled the moment retrieval is.
        abstention = abstention_for(sources)
        try:
            turn_id = conversations_repository.start_turn(conversation_id, query)
        except Exception:
            log.exception("/response could not commit the question for %s", filename)
            return jsonify({"error": "Internal server error"}), 500

        def event(name, payload):
            return f"event: {name}\ndata: {json.dumps(payload)}\n\n"

        if abstention is not None:
            # Nothing to answer from, and the app knows it before it spends
            # anything. The Turn is committed first, so the abstention is on
            # record exactly like an answer would be, and the browser reads it
            # through the same stream as every other outcome.
            log.info(
                "/response abstained for %s: %s (%s retrieved, %s usable)",
                filename,
                abstention.reason,
                abstention.retrieved,
                abstention.usable,
            )
            try:
                recorded = conversations_repository.abstain_turn(
                    turn_id, abstention.message, abstention.reason
                )
            except Exception:
                log.exception(
                    "/response could not record the abstention for %s", filename
                )
                return jsonify({"error": "Internal server error"}), 500
            if not recorded:
                log.warning("/response turn %s already ended for %s", turn_id, filename)
                return jsonify({"error": "Internal server error"}), 500

            def abstain():
                yield event("start", {"turn_id": turn_id})
                yield event(
                    "abstained", {**abstention.to_dict(), "retrieval": retrieval}
                )

            return Response(
                stream_with_context(abstain()),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        def done(answer_sources, resolved):
            """
            Return the terminal event for a stream that stored an answer.

            Only ever sent after ``complete_turn`` committed, so a browser that
            sees it knows the answer, its claims, and its Citation Sources are
            all in history.
            """
            reason = getattr(chat_provider, "last_finish_reason", None)
            # No finish reason at all means the provider never said the answer
            # was over — the app's own cap stopped it, or the stream ended
            # early. That is not proof the answer is whole, so it is reported
            # as unproven rather than as complete.
            truncated = reason not in model_definition.complete_finish_reasons
            return event(
                "done",
                {
                    "done": True,
                    "sources": answer_sources,
                    **resolved.citations.to_dict(),
                    "retrieval": retrieval,
                    "finish_reason": reason,
                    "truncated": truncated,
                },
            )

        def record_outcome(record, *args):
            """Apply one terminal turn outcome unless the document is gone."""
            try:
                if not _still_present():
                    log.warning(
                        "/response skipping turn outcome after concurrent delete for %s",
                        filename,
                    )
                    return
                record(*args)
            except Exception:
                log.exception(
                    "/response could not record turn outcome for %s", filename
                )

        # Each way an answer can end before it is stored gets its own event
        # and its own copy. A browser reads one terminal event and knows which
        # of these happened, instead of guessing from silence.
        provider_down = AnswerFailure(
            "Sorry. The language model is unavailable right now. Please try again.",
            "provider failure",
        )
        provider_slow = AnswerFailure(
            "This answer took too long and was stopped. Try asking a narrower question.",
            "provider timeout",
        )
        provider_silent = AnswerFailure(
            "The model returned nothing for this question. Try rephrasing it.",
            "empty answer",
        )
        unsaved = AnswerFailure(
            "The answer could not be saved. Please ask the question again.",
            "answer could not be saved",
        )
        uncited = AnswerFailure(
            "The answer cited a passage that was not supplied with the question, "
            "so it could not be shown. Please ask the question again.",
            UNRESOLVED_CITATIONS_REASON,
        )

        def provider_error(failure, category):
            """Close the turn as failed and return the terminal event."""
            record_outcome(
                conversations_repository.fail_turn,
                turn_id,
                failure.message,
                failure.reason,
            )
            return event(
                "provider_error", {"error": failure.message, "category": category}
            )

        def persistence_error(failure):
            """Close the turn as failed and return the terminal event."""
            record_outcome(
                conversations_repository.fail_turn,
                turn_id,
                failure.message,
                failure.reason,
            )
            return event("persistence_error", {"error": failure.message})

        def citation_error(failure):
            """Close the turn as failed and return the terminal event."""
            record_outcome(
                conversations_repository.fail_turn,
                turn_id,
                failure.message,
                failure.reason,
            )
            return event(
                "citation_error", {"error": failure.message, "category": "citations"}
            )

        def check_citations(generated, evidence) -> ResolvedAnswer | None:
            """
            Read one generated answer's claims and check every citation in them.

            Returns the answer prose beside its validated claims, or None when
            a citation still names nothing supplied after the one repair the
            app is willing to spend. A citation to a Passage the model was
            never shown is false, and a false citation that reaches the reader
            is worse than an answer that admits it could not be shown.
            """
            allowed = [source["source_id"] for source in evidence]
            parsed = parse_claims(generated)
            citations = validate_claims(
                prune_conflicting_claims(parsed.answer, parsed.claims), allowed
            )
            if not citations.invalid_ids:
                return ResolvedAnswer(parsed.answer, citations)
            log.warning(
                "/response cited passages not supplied for %s: %s of %s",
                filename,
                ",".join(citations.invalid_ids),
                ",".join(allowed),
            )
            # One repair, and only the mapping: the reader has already read the
            # answer, so a second version of the same sentences is a different
            # answer. A repair that cannot be read is as false as the original.
            repaired = parse_claims(
                chat_provider.generate_response(
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
                filename,
                ",".join(citations.invalid_ids),
            )
            return None

        def generate():
            yield event("start", {"turn_id": turn_id})
            fragments = []
            splitter = AnswerSplitter()
            try:
                for token in chat_provider.stream_response(
                    query, context, prior_turns_text
                ):
                    fragments.append(token)
                    # The claims block is written in the model's own output but
                    # belongs to the app, so it is collected and never shown.
                    visible = splitter.feed(token)
                    if visible:
                        yield event("token", {"text": visible})
                trailing = splitter.finish()
                if trailing:
                    yield event("token", {"text": trailing})
            except GeneratorExit:
                # The client left mid-answer. The question stays on record as
                # cancelled rather than pending, so it is never stranded, and
                # no fragment the provider still holds can reach a later
                # request.
                record_outcome(
                    conversations_repository.cancel_turn,
                    turn_id,
                    "client disconnected",
                )
                raise
            except ProviderTimeoutError as exc:
                log.warning("/response generation timed out for %s", filename)
                yield provider_error(provider_slow, "timeout")
                return
            except EmptyAnswerError as exc:
                log.warning("/response generation was empty for %s", filename)
                yield provider_error(provider_silent, "empty_output")
                return
            except Exception as exc:
                log.error(
                    "/response generation failed for %s: %s",
                    filename,
                    type(exc).__name__,
                )
                yield provider_error(provider_down, "provider")
                return

            answer = "".join(fragments).strip()
            if not _still_present():
                # The Document and its Conversation went away mid-answer, so
                # there is nothing left to store this in. Saying the answer was
                # saved would be a lie the browser would replay from history.
                # The Turn is closed as cancelled rather than left pending if
                # the delete has not reached it yet; a Turn that the delete
                # already took is a no-op here, not an error.
                log.warning(
                    "/response abandoning answer for deleted document %s", filename
                )
                try:
                    conversations_repository.cancel_turn(turn_id, "document deleted")
                except Exception:
                    log.exception(
                        "/response could not cancel turn for deleted document %s",
                        filename,
                    )
                yield event("cancelled", {"reason": "document deleted"})
                return
            citations = check_citations(answer, sources)
            if citations is None:
                yield citation_error(uncited)
                return
            try:
                completed = conversations_repository.complete_turn(
                    turn_id,
                    citations.answer or answer,
                    sources,
                    claims=[claim.to_dict() for claim in citations.citations.claims],
                )
            except Exception:
                log.exception("failed to persist answer for %s", filename)
                yield persistence_error(unsaved)
                return
            if not completed:
                # The Turn already reached a terminal state, so this answer is
                # not in history. A success event would put it on screen as if
                # it were.
                log.warning("/response turn %s already ended for %s", turn_id, filename)
                yield persistence_error(unsaved)
                return
            yield done(sources, citations)

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.route("/messages", methods=["GET"])
    def get_messages():
        try:
            filename = request.args.get("filename")
            if not filename:
                return jsonify({"error": "Filename is required"}), 400
            guard = traversal_check(services.storage, filename)
            if guard is not None:
                return guard
            file_record = files_repository.get_file(filename)
            if not file_record:
                return jsonify({"messages": []}), 200
            conversation_id = conversations_repository.get_conversation_id(
                file_record["id"]
            )
            if not conversation_id:
                return jsonify({"messages": []}), 200
            messages = conversations_repository.get_messages(conversation_id)
            return jsonify({"messages": messages}), 200
        except Exception:
            log.exception("get_messages failed for %r", request.args.get("filename"))
            return jsonify({"error": "Internal server error"}), 500
