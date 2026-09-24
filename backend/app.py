"""Flask application entry point."""

import logging

from flask import Flask, jsonify, request
from flask_cors import CORS

import settings
from composition import Services
from routes import register_routes

log = logging.getLogger(__name__)


def create_app(
    app_settings: settings.Settings | None = None,
    services: Services | None = None,
) -> Flask:
    """Create the Flask app from validated settings and injected dependencies."""
    app_settings = app_settings or settings.get_settings()
    settings.validate(app_settings)
    services = services or Services.from_settings(app_settings)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = app_settings.upload.max_upload_bytes
    origins = [
        origin.strip()
        for origin in app_settings.frontend.frontend_origin.split(",")
        if origin.strip()
    ]
    CORS(app, origins=origins, supports_credentials=False)

    @app.route("/health", methods=["GET"])
    def get_health():
        return jsonify({"response": "OK"}), 200

    register_routes(app, services)

    @app.errorhandler(413)
    def handle_413(_error):
        log.warning("request entity too large: %s", request.path)
        return jsonify({"error": "File too large"}), 413

    @app.errorhandler(500)
    def handle_500(_error):
        log.exception("internal server error for %s", request.path)
        return jsonify({"error": "Internal server error"}), 500

    return app


if __name__ == "__main__":
    create_app().run(debug=True, port=3000)
