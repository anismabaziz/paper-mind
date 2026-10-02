"""
Resolving the stored provider settings into a usable chat provider.

Every route that calls a model needs the same thing first: the stored App
Settings read, decrypted, checked against the catalog, and turned into
credentials and a bound provider. That sequence is where a missing key, an
incomplete row, and an unsupported model are caught, and each of them has to be
caught before a request reaches a model — so it lives here rather than in one
route, and both routes that call a model resolve it the same way.

The failure it raises is one of this project's own errors rather than an HTTP
response, so the same decision is reported in the same shape whichever route
made it, and neither of them invents its own wording for a broken key.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from errors import BadRequest
from services.accounts.chat_settings_service import (
    SUPPORTED_MODELS,
    ModelCapabilities,
    SettingsError,
    model_for,
)
from services.accounts.secrets_service import decrypt_api_key
from services.llm.base import ChatCredentials, LLMProvider

log = logging.getLogger(__name__)


#: The four ways stored settings can stop a request from reaching a model. Kept
#: as factories rather than as built errors because raising an exception attaches
#: the frame it was raised in to it, and one shared instance would hold that
#: traceback for every request it ever served.
def no_provider() -> BadRequest:
    """Return the error for there being no saved provider at all."""
    return BadRequest(
        "No chat provider configured. Add a provider and API key in Settings.",
        category="no_provider_configured",
    )


def incomplete_settings() -> BadRequest:
    """Return the error for a saved settings row that cannot be read."""
    return BadRequest(
        "Saved provider settings are incomplete. Re-save your provider "
        "settings in Settings.",
        category="incomplete_settings",
    )


def unsupported_provider() -> BadRequest:
    """Return the error for a saved provider this build does not serve."""
    return BadRequest(
        "Saved provider settings use an unsupported provider. Choose a current "
        "provider in Settings.",
        category="unsupported_provider",
    )


def unsupported_model() -> BadRequest:
    """Return the error for a saved model this build does not serve."""
    return BadRequest(
        "Saved provider settings use an unsupported model. Choose a current "
        "model in Settings.",
        category="unsupported_model",
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
) -> ProviderBinding:
    """
    Return the bound provider and its catalog record, or raise why there is none.

    Raises rather than returning a failure because every outcome here other than
    a usable provider is an error, and there is no route for which the error is
    not the answer. The messages live here so that a missing key reads the same
    whether it was hit asking a question or starting a brief.
    """
    stored = app_settings_repository.get_app_settings()
    if not stored:
        raise no_provider()
    try:
        provider = stored["provider"]
        model = stored["model"]
        ciphertext = stored["encrypted_api_key"]
    except (AttributeError, KeyError, TypeError):
        raise incomplete_settings() from None
    if (
        not isinstance(provider, str)
        or not provider
        or not isinstance(model, str)
        or not model
        or not isinstance(ciphertext, str)
        or not ciphertext
    ):
        raise incomplete_settings()
    # A key this build cannot decrypt is re-raised as-is: the central handler
    # already reports it as needing a re-save, which is the whole recovery.
    api_key = decrypt_api_key(ciphertext)
    if provider not in SUPPORTED_MODELS:
        raise unsupported_provider()
    try:
        model_definition = model_for(provider, model)
    except SettingsError:
        raise unsupported_model() from None
    credentials = ChatCredentials(
        provider=provider,
        model=model,
        api_key=api_key,
        verification_timeout_seconds=model_definition.timeout_seconds,
        budget=model_definition.chat_budget(),
    )
    try:
        bound = chat_provider_factory(credentials)
    except ValueError:
        log.warning("provider build refused for %s", provider)
        raise unsupported_provider() from None
    return ProviderBinding(provider=bound, model=model_definition)
