"""HTTP routes for global chat provider settings."""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from flask import Flask, jsonify, request

from services.accounts.chat_settings_service import (
    SUPPORTED_MODELS,
    SettingsError,
    mask_key,
    model_for,
    supported_models_payload,
    verification_error_message,
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


def _catalog_payload() -> dict[str, object]:
    """Return the public model catalog grouped by provider."""
    return {"supported_models": supported_models_payload()}


def _settings_payload(stored, plaintext_key=None):
    if not stored:
        return {
            "provider": None,
            "model": None,
            "masked_key": None,
            **_catalog_payload(),
        }
    provider = stored.get("provider")
    model = stored.get("model")
    ciphertext = stored.get("encrypted_api_key")
    if (
        not isinstance(provider, str)
        or not provider
        or not isinstance(model, str)
        or not model
        or not isinstance(ciphertext, str)
        or not ciphertext
    ):
        raise SettingsError("Saved provider settings are incomplete.")
    if provider not in SUPPORTED_MODELS or model not in SUPPORTED_MODELS[provider]:
        raise SettingsError(f"Saved provider settings use {model!r}.")
    if plaintext_key is None:
        plaintext_key = decrypt_api_key(ciphertext)
    return {
        "provider": provider,
        "model": model,
        "masked_key": mask_key(plaintext_key),
        **_catalog_payload(),
    }


def _candidate_from_request() -> ChatCredentials:
    """Parse and validate provider settings from a JSON request body."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise SettingsError("A JSON settings object is required")

    provider = data.get("provider")
    model = data.get("model")
    api_key = data.get("api_key")
    if not isinstance(provider, str) or not isinstance(model, str):
        raise SettingsError("Provider and model must be strings")
    if not isinstance(api_key, str) or not api_key.strip():
        raise SettingsError("API key is required")

    provider = provider.strip().lower()
    model = model.strip()
    api_key = api_key.strip()
    model_definition = model_for(provider, model)
    return ChatCredentials(
        provider=provider,
        model=model,
        api_key=api_key,
        verification_timeout_seconds=model_definition.timeout_seconds,
    )


def _error_response(message: str, status: int):
    """Return a settings error that keeps the model catalog available."""
    return jsonify({"error": message, **_catalog_payload()}), status


def _recovery_message(detail: str | None, has_saved_settings: bool | None) -> str:
    """Explain that a failed candidate did not replace saved settings."""
    if has_saved_settings is False:
        status = "Settings were not changed. No saved settings were affected."
    elif has_saved_settings is True:
        status = "Settings were not changed. Saved settings were left unchanged."
    else:
        status = "Settings were not changed."
    return (
        f"{status} Check the provider details and API key, then try again. "
        f"{detail or ''}"
    ).rstrip()


def _verify_candidate(
    credentials: ChatCredentials,
    verifier: Callable[[ChatCredentials], tuple[bool, str | None]],
) -> tuple[bool, str | None]:
    """Verify a candidate while keeping secrets out of responses and logs."""
    try:
        ok, error = verifier(credentials)
    except Exception as exc:
        error = verification_error_message(exc)
        log.error("candidate verification failed: %s", error)
        ok = False
    if error is not None:
        error = verification_error_message(error)
        ok = False
    return bool(ok), error


def _clear_llm_caches() -> None:
    try:
        from services.llm import google_provider, groq_provider
    except Exception as exc:
        log.warning(
            "failed to load LLM cache modules: %s",
            type(exc).__name__,
        )
        return

    for provider_cache in (groq_provider.clear_cache, google_provider.clear_cache):
        try:
            provider_cache()
        except Exception as exc:
            log.warning(
                "failed to clear an LLM cache: %s",
                type(exc).__name__,
            )


def register_settings_routes(app: Flask, services: "Services") -> None:
    """Register global chat settings routes."""
    app_settings_repository = services.repositories.app_settings
    api_key_verifier = services.api_key_verifier

    @app.route("/settings", methods=["GET"])
    def get_settings_route():
        try:
            stored = app_settings_repository.get_app_settings()
        except Exception as exc:
            log.error("get_settings failed: %s", type(exc).__name__)
            return _error_response(
                "Stored settings could not be read. Re-save your provider "
                "settings, then try again.",
                500,
            )
        try:
            payload = _settings_payload(stored)
        except SettingsError as exc:
            provider = stored.get("provider") if isinstance(stored, dict) else None
            model = stored.get("model") if isinstance(stored, dict) else None
            return jsonify(
                {
                    "provider": provider,
                    "model": model,
                    "masked_key": None,
                    "error": str(exc),
                    **_catalog_payload(),
                    "needs_resave": True,
                }
            ), 400
        except SecretsResaveRequiredError as exc:
            log.warning("get_settings stale key derivation")
            return jsonify(
                {
                    "error": str(exc),
                    **_catalog_payload(),
                    "needs_resave": True,
                }
            ), 400
        except Exception as exc:
            log.error("get_settings failed: %s", type(exc).__name__)
            return _error_response(
                "Stored settings could not be read. Re-save your provider "
                "settings, then try again.",
                500,
            )
        return jsonify(payload), 200

    @app.route("/settings", methods=["PUT"])
    def save_settings():
        try:
            credentials = _candidate_from_request()
        except SettingsError as exc:
            return _error_response(_recovery_message(str(exc), None), 400)
        try:
            stored = app_settings_repository.get_app_settings()
        except Exception as exc:
            log.error("save_settings read failed: %s", type(exc).__name__)
            return _error_response(
                "Stored settings could not be read. Re-save your provider "
                "settings, then try again.",
                500,
            )
        has_saved_settings = stored is not None

        ok, error = _verify_candidate(credentials, api_key_verifier)
        if not ok:
            return _error_response(_recovery_message(error, has_saved_settings), 400)

        try:
            encrypted = encrypt_api_key(credentials.api_key)
        except Exception as exc:
            log.error("encrypt_api_key failed: %s", type(exc).__name__)
            return _error_response(
                _recovery_message(
                    "The candidate key could not be encrypted.", has_saved_settings
                ),
                500,
            )
        try:
            app_settings_repository.upsert_app_settings(
                credentials.provider,
                credentials.model,
                encrypted,
            )
        except Exception as exc:
            log.error("upsert_app_settings failed: %s", type(exc).__name__)
            return _error_response(
                _recovery_message(
                    "The verified settings could not be stored. Try again.",
                    has_saved_settings,
                ),
                500,
            )
        _clear_llm_caches()
        return jsonify(
            {
                "provider": credentials.provider,
                "model": credentials.model,
                "masked_key": mask_key(credentials.api_key),
                **_catalog_payload(),
            }
        ), 200

    @app.route("/settings/verify", methods=["POST"])
    def verify_settings():
        try:
            credentials = _candidate_from_request()
        except SettingsError as exc:
            return jsonify(
                {
                    "ok": False,
                    "error": _recovery_message(str(exc), None),
                    **_catalog_payload(),
                }
            ), 400
        try:
            stored = app_settings_repository.get_app_settings()
        except Exception as exc:
            log.error("verify_settings read failed: %s", type(exc).__name__)
            return jsonify(
                {
                    "ok": False,
                    "error": "Stored settings could not be read.",
                    **_catalog_payload(),
                }
            ), 500
        has_saved_settings = stored is not None
        ok, error = _verify_candidate(credentials, api_key_verifier)
        if not ok:
            error = _recovery_message(error, has_saved_settings)
        return jsonify({"ok": ok, "error": error, **_catalog_payload()}), 200
