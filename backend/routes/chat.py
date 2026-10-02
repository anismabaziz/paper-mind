"""HTTP routes for document chat and message history."""

import logging
from typing import TYPE_CHECKING

from flask import Flask, Response, jsonify, request, stream_with_context

from errors import refusal
from routes.common import check_filename, raise_refusal
from routes.credentials import resolve_stored_provider
from services.answering import AnswerRequest, AnswerService

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
        vector_service=services.vector_service,
        tracer=services.tracer,
    )

    @app.route("/response", methods=["POST"])
    def get_response():
        data = request.get_json() or {}
        query = data.get("query")
        filename = data.get("filename")
        if not query or not filename:
            # Named the way the answering service names the same refusal, so a
            # caller that cannot ask sees one category whether the question was
            # rejected before it reached the model or while being resolved.
            raise refusal("Query and Filename are required", "invalid_query")

        check_filename(services.storage, filename)

        binding = resolve_stored_provider(
            app_settings_repository, chat_provider_factory
        )

        resolved = raise_refusal(
            answer_service.resolve(
                AnswerRequest(
                    filename=filename,
                    query=query,
                    provider=binding.provider,
                    model=binding.model,
                )
            )
        )

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
        filename = request.args.get("filename")
        if not filename:
            raise refusal("Filename is required")
        check_filename(services.storage, filename)
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
