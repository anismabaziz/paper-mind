"""HTTP routes for document chat and message history."""

import json
import logging
from typing import TYPE_CHECKING

from flask import Flask, Response, jsonify, request, stream_with_context

from routes.common import traversal_check, vector_store_error_response
from services.accounts.chat_settings_service import (
    SUPPORTED_MODELS,
    SettingsError,
    model_for,
)
from services.accounts.secrets_service import (
    SecretsResaveRequiredError,
    decrypt_api_key,
)
from services.llm.base import ChatCredentials
from services.retrieval.base import (
    VectorDimensionError,
    VectorStoreConfigurationError,
    VectorStoreUnavailableError,
)

if TYPE_CHECKING:
    from composition import Services

log = logging.getLogger(__name__)

MAX_QUERY_CHARS = 8192


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
        conversation_id = conversations_repository.get_conversation_id(
            file_record["id"]
        )
        if not conversation_id:
            return jsonify({"error": "Conversation not found"}), 404
        if not isinstance(query, str) or not query.strip():
            return jsonify({"error": "Query and Filename are required"}), 400
        if len(query) > MAX_QUERY_CHARS:
            return jsonify({"error": "Query is too long"}), 400

        try:
            query_embedding = embedding_service.embed_texts(query)[0]
            retrieval_result = vector_service.query_vectors(
                query_embedding, filename, query_text=query
            )
            retrieval = {
                "method": retrieval_result.method,
                "outcome": retrieval_result.outcome,
            }
            sources = [_normalize_source(source) for source in retrieval_result.sources]
            context = "\n\n".join(source["content"] for source in sources)
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
        try:
            conversations_repository.add_message(conversation_id, "user", query)
        except Exception:
            log.exception("/response persist user message failed for %s", filename)
            return jsonify({"error": "Internal server error"}), 500

        def event(name, payload):
            return f"event: {name}\ndata: {json.dumps(payload)}\n\n"

        def generate():
            fragments = []
            try:
                for token in chat_provider.stream_response(query, context):
                    fragments.append(token)
                    yield event("token", {"text": token})
            except Exception as exc:
                log.error(
                    "/response generation failed for %s: %s",
                    filename,
                    type(exc).__name__,
                )
                failure = (
                    "Sorry. The language model is unavailable right now. "
                    "Please try again."
                )
                try:
                    conversations_repository.add_message(
                        conversation_id, "bot", failure
                    )
                except Exception:
                    log.exception("failed to persist error reply for %s", filename)
                yield event("error", {"error": failure})
                yield event(
                    "done", {"done": True, "sources": [], "retrieval": retrieval}
                )
                return

            answer = (
                "".join(fragments).strip() or "I don't know based on the given context."
            )
            try:
                conversations_repository.add_message(
                    conversation_id, "bot", answer, sources
                )
            except Exception:
                log.exception("failed to persist answer for %s", filename)
            yield event(
                "done",
                {"done": True, "sources": sources, "retrieval": retrieval},
            )

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
