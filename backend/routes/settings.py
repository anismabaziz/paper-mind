"""HTTP routes for global chat provider settings."""

import logging
from typing import TYPE_CHECKING

from flask import Flask, jsonify, request

from routes.common import scrub_api_key_from_text
from services.accounts.chat_settings_service import (
    SUPPORTED_MODELS,
    SettingsError,
    mask_key,
    validate,
)
from services.accounts.secrets_service import (
    SecretsResaveRequiredError,
    decrypt_api_key,
    encrypt_api_key,
)
from services.llm.base import ChatCredentials

if TYPE_CHECKING:
    from composition import Services

log = logging.getLogger(__name__)


def _settings_payload(stored, plaintext_key=None):
    if not stored:
        return {
            "provider": None,
            "model": None,
            "masked_key": None,
            "supported_models": SUPPORTED_MODELS,
        }
    if plaintext_key is None:
        plaintext_key = decrypt_api_key(stored["encrypted_api_key"])
    return {
        "provider": stored["provider"],
        "model": stored["model"],
        "masked_key": mask_key(plaintext_key),
        "supported_models": SUPPORTED_MODELS,
    }


def _clear_llm_caches() -> None:
    try:
        from services.llm import google_provider, groq_provider

        groq_provider.clear_cache()
        google_provider.clear_cache()
    except Exception as exc:
        log.warning("failed to clear LLM caches: %s", exc)


def register_settings_routes(app: Flask, services: "Services") -> None:
    """Register global chat settings routes."""
    app_settings_repository = services.repositories.app_settings
    api_key_verifier = services.api_key_verifier

    @app.route("/settings", methods=["GET"])
    def get_settings_route():
        stored = app_settings_repository.get_app_settings()
        try:
            payload = _settings_payload(stored)
        except SecretsResaveRequiredError as exc:
            log.warning("get_settings stale key derivation")
            return jsonify({"error": str(exc)}), 400
        except Exception:
            log.exception("get_settings failed")
            return jsonify({"error": "Internal server error"}), 500
        return jsonify(payload), 200

    @app.route("/settings", methods=["PUT"])
    def save_settings():
        data = request.get_json() or {}
        provider = (data.get("provider") or "").strip().lower()
        model = (data.get("model") or "").strip()
        api_key = data.get("api_key") or ""
        if not api_key:
            return jsonify({"error": "API key is required"}), 400
        try:
            validate(provider, model)
        except SettingsError as exc:
            return jsonify({"error": str(exc)}), 400
        try:
            encrypted = encrypt_api_key(api_key)
        except Exception:
            log.exception("encrypt_api_key failed")
            return jsonify({"error": "Internal server error"}), 500
        try:
            app_settings_repository.upsert_app_settings(provider, model, encrypted)
        except Exception:
            log.exception("upsert_app_settings failed")
            return jsonify({"error": "Internal server error"}), 500
        _clear_llm_caches()
        return jsonify(
            {
                "provider": provider,
                "model": model,
                "masked_key": mask_key(api_key),
                "supported_models": SUPPORTED_MODELS,
            }
        ), 200

    @app.route("/settings/verify", methods=["POST"])
    def verify_settings():
        stored = app_settings_repository.get_app_settings()
        if not stored:
            return jsonify({"error": "No chat settings saved yet"}), 400
        try:
            api_key = decrypt_api_key(stored["encrypted_api_key"])
        except SecretsResaveRequiredError as exc:
            log.warning("verify stale key derivation")
            return jsonify({"error": str(exc)}), 400
        except Exception:
            log.exception("verify decrypt failed")
            return jsonify({"error": "Internal server error"}), 500
        credentials = ChatCredentials(
            provider=stored["provider"],
            model=stored["model"],
            api_key=api_key,
        )
        ok, error = api_key_verifier(credentials)
        if error:
            error = scrub_api_key_from_text(error, api_key) or "Verification failed"
            _clear_llm_caches()
        return jsonify({"ok": ok, "error": error}), 200
