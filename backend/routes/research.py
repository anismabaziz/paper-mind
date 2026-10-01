"""HTTP routes for the Research Brief: a bounded run over two Documents."""

import logging
from typing import TYPE_CHECKING

from flask import Flask, Response, jsonify, request, stream_with_context

from routes.common import traversal_check
from routes.credentials import resolve_stored_provider
from services.answering import Refusal
from services.brief.scope import resolve_scope
from services.brief import cancellation
from services.brief.budget import BriefLimits
from services.brief.service import BriefRequest, BriefService

if TYPE_CHECKING:
    from composition import Services

log = logging.getLogger(__name__)


def _brief_limits(settings) -> BriefLimits:
    """Return the brief's spend ceilings as the running configuration sets them."""
    configured = settings.research
    return BriefLimits(
        max_turns=configured.max_turns,
        max_tool_calls=configured.max_tool_calls,
        max_repeated_calls=configured.max_repeated_calls,
        max_tokens=configured.max_tokens,
        max_seconds=configured.max_seconds,
    )


def register_research_routes(app: Flask, services: "Services") -> None:
    """Register the Research Brief route over the injected application graph."""
    brief_service = BriefService(
        settings=services.settings,
        repositories=services.repositories,
        embedding_service=services.embedding_service,
        vector_service=services.vector_service,
        tracer=services.tracer,
        storage=services.storage,
        parser=services.parser,
    )

    @app.route("/research", methods=["POST"])
    def start_research_brief():
        """
        Start one Research Brief over exactly two indexed Documents.

        The pair, the model, and the ceilings are all decided before the first
        model turn, so a refusal here costs nothing and a stream that opens is
        one that will run. The response is a server-sent event stream: the brief
        takes several turns and its evidence arrives as the model finds it, and
        a reader who waits for a single JSON body would be watching nothing for
        the whole run.
        """
        data = request.get_json(silent=True) or {}
        question = data.get("question")
        documents = data.get("documents")
        if not isinstance(question, str) or not question.strip():
            return jsonify({"error": "A research question is required"}), 400
        if not isinstance(documents, list) or not all(
            isinstance(name, str) for name in documents
        ):
            return jsonify({"error": "Two Documents are required"}), 400
        for filename in documents:
            guard = traversal_check(services.storage, filename)
            if guard is not None:
                return guard

        # Read per request rather than captured at boot: a ceiling the operator
        # changes should apply to the next brief, not to the next restart.
        limits = _brief_limits(services.settings)
        scope, scope_refusal = resolve_scope(
            repositories=services.repositories,
            settings=services.settings,
            filenames=list(documents),
            limit=limits.documents,
        )
        if scope is None:
            refusal = _scope_refusal(scope_refusal)
            return jsonify(refusal.to_dict()), refusal.status

        binding = resolve_stored_provider(
            services.repositories.app_settings, services.chat_provider_factory
        )
        if isinstance(binding, Refusal):
            return jsonify(binding.to_dict()), binding.status

        resolved = brief_service.resolve(
            BriefRequest(
                filenames=scope.filenames,
                question=question,
                provider=binding.provider,
                model=binding.model,
                limits=limits,
            ),
            scope,
        )
        if isinstance(resolved, Refusal):
            return jsonify(resolved.to_dict()), resolved.status

        return Response(
            stream_with_context(
                event.as_server_sent_event() for event in brief_service.stream(resolved)
            ),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.route("/research/cancel", methods=["POST"])
    def cancel_research_brief():
        """
        Ask a running brief to stop, so the reader keeps what it found.

        Stopping by closing the stream would abandon the brief along with the
        connection, which is right for someone who left the page and wrong for
        someone who pressed stop. This asks the loop to finish normally instead:
        it stops between turns, sends a terminal event with the evidence it had
        collected, and marks the result incomplete.
        """
        data = request.get_json(silent=True) or {}
        brief_id = data.get("brief_id")
        if not isinstance(brief_id, str) or not brief_id:
            return jsonify({"error": "A brief id is required"}), 400
        stopped = cancellation.cancel(brief_id)
        if not stopped:
            return (
                jsonify(
                    {
                        "error": "That brief is not running.",
                        "category": "brief_not_running",
                    }
                ),
                409,
            )
        return jsonify({"brief_id": brief_id, "cancelled": True}), 200

    @app.route("/research/scope", methods=["GET"])
    def research_brief_scope():
        """
        Return which Documents a brief could be run over right now.

        The interface asks this instead of guessing from the library listing:
        a Document being reindexed, being removed, or carrying an index that no
        longer matches the running configuration cannot be part of a brief, and
        the listing does not say so. Whether the chosen model can run one at all
        is a separate question the interface already answers from App Settings,
        which publish each catalog model's tool and structured-output support.
        """
        filenames = request.args.getlist("documents")
        limits = _brief_limits(services.settings)
        scope, scope_refusal = resolve_scope(
            repositories=services.repositories,
            settings=services.settings,
            filenames=list(filenames),
            limit=limits.documents,
        )
        if scope is None:
            refusal = _scope_refusal(scope_refusal)
            return jsonify(refusal.to_dict()), refusal.status
        return jsonify(
            {"documents": scope.to_dict(), "document_count": limits.documents}
        ), 200


def _scope_refusal(detail: dict | None) -> Refusal:
    """Return the refusal a rejected scope is reported as."""
    detail = detail or {}
    return Refusal(
        int(detail.get("status", 400)),
        str(detail.get("category", "")),
        str(detail.get("error", "Internal server error")),
        {
            key: value
            for key, value in detail.items()
            if key not in ("status", "category", "error")
        },
    )
