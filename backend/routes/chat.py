"""HTTP routes for document chat and message history."""

import logging
from typing import TYPE_CHECKING

from flask import Flask, Response, jsonify, request, stream_with_context

from routes.common import traversal_check
from routes.credentials import resolve_stored_provider
from services.answering import AnswerRequest, AnswerService, Refusal

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
        tracer=services.tracer,
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

        binding = resolve_stored_provider(
            app_settings_repository, chat_provider_factory
        )
        if isinstance(binding, Refusal):
            return jsonify(binding.to_dict()), binding.status

        resolved = answer_service.resolve(
            AnswerRequest(
                filename=filename,
                query=query,
                provider=binding.provider,
                model=binding.model,
            )
        )
        if isinstance(resolved, Refusal):
            return jsonify(resolved.to_dict()), resolved.status

        return Response(
            stream_with_context(
                event.as_server_sent_event()
                for event in answer_service.stream(resolved)
            ),
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
