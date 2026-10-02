"""HTTP routes for global chat provider settings."""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from flask import Flask, jsonify, request

from errors import AppError
from services.accounts.chat_settings_service import (
    SUPPORTED_MODELS,
    SettingsError,
    mask_key,
    model_for,
    supported_models_payload,
    verification_error_message,
)
from services.accounts.secrets_service import decrypt_api_key, encrypt_api_key
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


def _settings_error(
    message: str,
    category: str,
    status: int = 400,
    **details: object,
) -> AppError:
    """
    Return a settings failure carrying the model catalog.

    Every settings failure carries the catalog because the settings form is
    built from it: a failure that returned none would leave someone unable to
    correct the thing that just failed. It rides in the details rather than being
    restated in each message, so the catalog is sent the same way whatever went
    wrong.
    """
    return AppError(
        message,
        category=category,
        status=status,
        details={**details, **_catalog_payload()},
    )


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

    def _read_stored():
        """
        Return the saved settings row, or raise why it could not be read.

        The catalog goes out with the failure rather than being lost, because a
        settings form that cannot be built is exactly what someone hitting this
        needs to be spared: the stored settings are unreadable, but the catalog of
        models they could choose from is not.
        """
        try:
            return app_settings_repository.get_app_settings()
        except Exception as exc:
            log.error("reading stored settings failed: %s", type(exc).__name__)
            raise _settings_error(
                "Stored settings could not be read. Re-save your provider "
                "settings, then try again.",
                "settings_unreadable",
                500,
            ) from None

    @app.route("/settings", methods=["GET"])
    def get_settings_route():
        stored = _read_stored()
        try:
            payload = _settings_payload(stored)
        except SettingsError as exc:
            # The saved provider and model are echoed back because the row is
            # still readable even though it cannot be used, and the form needs
            # to show what is saved to let the reader replace it. A key this
            # build cannot decrypt is not caught here: it already carries the
            # re-save the reader needs, and the catalog rides along with it.
            record = stored if isinstance(stored, dict) else {}
            raise _settings_error(
                str(exc),
                "incomplete_settings",
                400,
                provider=record.get("provider"),
                model=record.get("model"),
                masked_key=None,
                needs_resave=True,
            ) from None
        return jsonify(payload), 200

    @app.route("/settings", methods=["PUT"])
    def save_settings():
        try:
            credentials = _candidate_from_request()
        except SettingsError as exc:
            raise _settings_error(
                _recovery_message(str(exc), None), "invalid_settings"
            ) from None

        has_saved_settings = _read_stored() is not None

        ok, error = _verify_candidate(credentials, api_key_verifier)
        if not ok:
            raise _settings_error(
                _recovery_message(error, has_saved_settings),
                "settings_verification_failed",
            )

        try:
            encrypted = encrypt_api_key(credentials.api_key)
            app_settings_repository.upsert_app_settings(
                credentials.provider,
                credentials.model,
                encrypted,
            )
        except Exception as exc:
            # The candidate verified but could not be stored, so the app is left
            # exactly as it was. Said plainly, because a reader who assumed the
            # save went through would go on using the old settings.
            log.error("storing verified settings failed: %s", type(exc).__name__)
            raise _settings_error(
                _recovery_message(
                    "The verified settings could not be stored. Try again.",
                    has_saved_settings,
                ),
                "settings_not_stored",
                500,
            ) from None

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
            raise _settings_error(
                _recovery_message(str(exc), None),
                "invalid_settings",
                ok=False,
            ) from None

        has_saved_settings = _read_stored() is not None
        ok, error = _verify_candidate(credentials, api_key_verifier)
        # A key that does not work is an answer, not a failure of the request: the
        # request was understood and answered, and the answer is that the key was
        # refused. A refused key therefore comes back in the same three keys every
        # other failure uses, beside `ok`, so a caller reads one failure shape.
        # A key that works is not a failure at all and carries no failure keys —
        # only the reasons it could not be answered come back as errors.
        if ok:
            return (
                jsonify({"ok": True, **_catalog_payload()}),
                200,
            )
        raise _settings_error(
            _recovery_message(error, has_saved_settings),
            "settings_verification_failed",
            400,
            ok=False,
        )
