"""HTTP routes for document chat and message history."""

import json
import logging
from typing import TYPE_CHECKING

from flask import Flask, Response, jsonify, request, stream_with_context

from routes.common import traversal_check
from services.accounts.chat_settings_service import (
    SUPPORTED_MODELS,
    SettingsError,
    model_for,
)
from services.accounts.secrets_service import (
    SecretsResaveRequiredError,
    decrypt_api_key,
)
from services.answering import AnswerRequest, AnswerService, Refusal
from services.llm.base import ChatCredentials

if TYPE_CHECKING:
    from composition import Services

log = logging.getLogger(__name__)


def register_chat_routes(app: Flask, services: "Services") -> None:
    """Register chat response and message-history routes."""
    app_settings_repository = services.repositories.app_settings
    files_repository = services.repositories.files
    conversations_repository = services.repositories.conversations
    chat_provider_factory = services.chat_provider_factory
    answer_service = AnswerService(
        settings=services.settings,
        repositories=services.repositories,
        embedding_service=services.embedding_service,
        vector_service=services.vector_service,
    )

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

        def refusal_response(refusal: Refusal):
            """Return the HTTP response for a question that was not answered."""
            return jsonify(refusal.to_dict()), refusal.status

        def stream_response(answer_service, resolved):
            """Yield the answer path's events as server-sent events."""
            for event in answer_service.stream(resolved):
                yield (f"event: {event.name}\ndata: {json.dumps(event.payload)}\n\n")

        resolved = answer_service.resolve(
            AnswerRequest(
                filename=filename,
                query=query,
                provider=chat_provider,
                model=model_definition,
            )
        )
        if isinstance(resolved, Refusal):
            return refusal_response(resolved)

        return Response(
            stream_with_context(stream_response(answer_service, resolved)),
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
