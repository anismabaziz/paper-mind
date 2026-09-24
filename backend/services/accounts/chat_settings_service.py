"""
Global chat settings.

Supported model catalog, validation, key masking, and stored-key
verification for the singleton ``app_settings`` row. Routes stay thin;
the catalog and provider calls live here so adding a model means editing
one list.
"""

from dataclasses import asdict, dataclass
from typing import Literal

from services.llm.base import VERIFY_TIMEOUT_SECONDS, ChatCredentials


@dataclass(frozen=True)
class ModelCapabilities:
    """Published capabilities for one supported chat model."""

    provider: str
    id: str
    context_window_tokens: int
    max_output_tokens: int
    structured_output: bool
    tool_use: bool
    input_cost_per_million_usd: float
    output_cost_per_million_usd: float
    pricing_tier: Literal["standard"]
    data_location: Literal["cloud", "local"]
    timeout_seconds: float


MODEL_CATALOG = (
    ModelCapabilities(
        provider="google",
        id="gemini-2.5-flash",
        context_window_tokens=1_048_576,
        max_output_tokens=65_536,
        structured_output=True,
        tool_use=True,
        input_cost_per_million_usd=0.30,
        output_cost_per_million_usd=2.50,
        pricing_tier="standard",
        data_location="cloud",
        timeout_seconds=VERIFY_TIMEOUT_SECONDS,
    ),
    ModelCapabilities(
        provider="google",
        id="gemini-3.5-flash",
        context_window_tokens=1_048_576,
        max_output_tokens=65_536,
        structured_output=True,
        tool_use=True,
        input_cost_per_million_usd=1.50,
        output_cost_per_million_usd=9.00,
        pricing_tier="standard",
        data_location="cloud",
        timeout_seconds=VERIFY_TIMEOUT_SECONDS,
    ),
    ModelCapabilities(
        provider="groq",
        id="openai/gpt-oss-120b",
        context_window_tokens=131_072,
        max_output_tokens=65_536,
        structured_output=True,
        tool_use=True,
        input_cost_per_million_usd=0.15,
        output_cost_per_million_usd=0.60,
        pricing_tier="standard",
        data_location="cloud",
        timeout_seconds=VERIFY_TIMEOUT_SECONDS,
    ),
    ModelCapabilities(
        provider="groq",
        id="openai/gpt-oss-20b",
        context_window_tokens=131_072,
        max_output_tokens=65_536,
        structured_output=True,
        tool_use=True,
        input_cost_per_million_usd=0.075,
        output_cost_per_million_usd=0.30,
        pricing_tier="standard",
        data_location="cloud",
        timeout_seconds=VERIFY_TIMEOUT_SECONDS,
    ),
)

DEFAULT_PROVIDER = "google"
DEFAULT_MODEL = "gemini-2.5-flash"

SUPPORTED_MODELS = {
    provider: tuple(model.id for model in MODEL_CATALOG if model.provider == provider)
    for provider in sorted({model.provider for model in MODEL_CATALOG})
}
SUPPORTED_MODEL_DEFINITIONS = {
    provider: tuple(model for model in MODEL_CATALOG if model.provider == provider)
    for provider in SUPPORTED_MODELS
}


def supported_models_payload() -> dict[str, list[dict[str, object]]]:
    """Return the model catalog grouped by provider."""
    return {
        provider: [asdict(model) for model in models]
        for provider, models in SUPPORTED_MODEL_DEFINITIONS.items()
    }


class SettingsError(Exception):
    """SettingsError."""

    pass


def validate(provider: str, model: str) -> None:
    """Raise SettingsError unless the provider/model pair is supported."""
    if provider not in SUPPORTED_MODELS:
        raise SettingsError(
            f"Unsupported provider {provider!r}; expected one of "
            f"{sorted(SUPPORTED_MODELS)}"
        )
    if model not in SUPPORTED_MODELS[provider]:
        raise SettingsError(f"Unsupported model {model!r} for provider {provider!r}")


def model_for(provider: str, model: str) -> ModelCapabilities:
    """Return the capability record for a supported provider/model pair."""
    validate(provider, model)
    return next(
        entry
        for entry in MODEL_CATALOG
        if entry.provider == provider and entry.id == model
    )


def verification_error_message(error: object) -> str:
    """Return a stable, key-free message for a verification failure."""
    text = str(error).lower() if error is not None else ""
    if any(
        marker in text
        for marker in (
            "api key",
            "invalid key",
            "unauthorized",
            "unauthorised",
            "authentication",
            "permission",
            "forbidden",
            "401",
            "403",
        )
    ):
        return "The provider rejected the API key. Check the key, then try again."
    if any(
        marker in text
        for marker in (
            "timeout",
            "timed out",
            "deadline",
            "temporarily",
            "unavailable",
            "connection",
            "network",
            "500",
            "502",
            "503",
        )
    ):
        return "The provider did not respond. Try again."
    if any(
        marker in text
        for marker in ("model", "not supported", "not found", "404", "unknown")
    ):
        return "The provider could not use this model. Choose a supported model."
    return "The provider could not verify these settings. Check the values."


def mask_key(api_key: str) -> str:
    """Return a display-safe mask that keeps only the last four characters."""
    if len(api_key) <= 4:
        return "••••"
    return f"••••{api_key[-4:]}"


def verify_api_key(credentials: ChatCredentials) -> tuple[bool, str | None]:
    """Run a one-token completion; return (ok, error_message)."""
    from services.llm.factory import build_chat_provider

    try:
        build_chat_provider(credentials, use_cache=False).verify()
        return True, None
    except Exception as e:
        return False, verification_error_message(e)
