"""Flask application entry point."""

import logging

from flask import Flask, jsonify
from flask_cors import CORS

import settings
from composition import Services
from errors import register_domain_errors, register_error_handlers
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

    # Registered before the routes so that a route raising one of the domain's
    # own errors is reported through the central handler rather than by
    # whichever route happened to anticipate it.
    register_domain_errors()

    app = Flask(__name__)
    register_error_handlers(app)
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

    return app


if __name__ == "__main__":
    create_app().run(debug=True, port=3000)
