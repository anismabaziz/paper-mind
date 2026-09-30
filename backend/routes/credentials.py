"""
Resolving the stored provider settings into a usable chat provider.

Every route that calls a model needs the same thing first: the stored App
Settings read, decrypted, checked against the catalog, and turned into
credentials and a bound provider. That sequence is where a missing key, an
incomplete row, and an unsupported model are caught, and each of them has to be
caught before a request reaches a model — so it lives here rather than in one
route, and both routes that call a model resolve it the same way.

The failure it returns is a :class:`~services.answering.Refusal` rather than an
HTTP response, so the same decision can be reported as a refusal from either
route and neither of them invents its own wording for a broken key.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from services.accounts.chat_settings_service import (
    SUPPORTED_MODELS,
    ModelCapabilities,
    SettingsError,
    model_for,
)
from services.accounts.secrets_service import (
    SecretsResaveRequiredError,
    decrypt_api_key,
)
from services.answering import Refusal
from services.llm.base import ChatCredentials, LLMProvider

log = logging.getLogger(__name__)

NO_PROVIDER = Refusal(
    400,
    "no_provider_configured",
    "No chat provider configured. Add a provider and API key in Settings.",
)
INCOMPLETE_SETTINGS = Refusal(
    400,
    "incomplete_settings",
    "Saved provider settings are incomplete. Re-save your provider settings in "
    "Settings.",
)
UNSUPPORTED_PROVIDER = Refusal(
    400,
    "unsupported_provider",
    "Saved provider settings use an unsupported provider. Choose a current "
    "provider in Settings.",
)
UNSUPPORTED_MODEL = Refusal(
    400,
    "unsupported_model",
    "Saved provider settings use an unsupported model. Choose a current model "
    "in Settings.",
)
KEY_UNREADABLE = Refusal(
    500,
    "",
    "Stored API key could not be decrypted. Re-save your provider settings, "
    "then try again.",
)


def _resave_required(message: str) -> Refusal:
    """
    Return the refusal for a key this build can no longer decrypt.

    The client is told so explicitly, because the recovery is a re-save and
    nothing else: retrying unchanged fails the same way every time. The wording
    comes from the failure itself, which is what tells a user whether their key
    was encrypted by an older version or is simply not there.
    """
    return Refusal(
        400,
        "secrets_resave_required",
        message,
        {"needs_resave": True},
    )


@dataclass(frozen=True)
class ProviderBinding:
    """
    The provider a request will call, and the catalog entry describing it.

    Both travel together because a caller needs both, and reading the catalog
    entry again from the settings row would be a second source of truth for
    which model is running.
    """

    provider: LLMProvider
    model: ModelCapabilities


def resolve_stored_provider(
    app_settings_repository: Any,
    chat_provider_factory: Any,
) -> ProviderBinding | Refusal:
    """Return the bound provider and its catalog record, or the refusal instead."""
    try:
        stored = app_settings_repository.get_app_settings()
    except Exception as exc:
        log.error("stored settings read failed: %s", type(exc).__name__)
        return Refusal(500, "", "Stored settings could not be read.")
    if not stored:
        return NO_PROVIDER
    try:
        provider = stored["provider"]
        model = stored["model"]
        ciphertext = stored["encrypted_api_key"]
    except (AttributeError, KeyError, TypeError):
        return INCOMPLETE_SETTINGS
    if (
        not isinstance(provider, str)
        or not provider
        or not isinstance(model, str)
        or not model
        or not isinstance(ciphertext, str)
        or not ciphertext
    ):
        return INCOMPLETE_SETTINGS
    try:
        api_key = decrypt_api_key(ciphertext)
    except SecretsResaveRequiredError as exc:
        log.warning("stale key derivation")
        return _resave_required(str(exc))
    except Exception as exc:
        log.error("stored key decrypt failed: %s", type(exc).__name__)
        return KEY_UNREADABLE
    if provider not in SUPPORTED_MODELS:
        return UNSUPPORTED_PROVIDER
    try:
        model_definition = model_for(provider, model)
    except SettingsError:
        return UNSUPPORTED_MODEL
    credentials = ChatCredentials(
        provider=provider,
        model=model,
        api_key=api_key,
        verification_timeout_seconds=model_definition.timeout_seconds,
        budget=model_definition.chat_budget(),
    )
    try:
        bound = chat_provider_factory(credentials)
    except ValueError as exc:
        log.warning("provider build refused: %s", type(exc).__name__)
        return UNSUPPORTED_PROVIDER
    except Exception as exc:
        log.error("provider build failed: %s", type(exc).__name__)
        return Refusal(500, "", "Internal server error")
    return ProviderBinding(provider=bound, model=model_definition)
