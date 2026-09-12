"""
Global chat settings.

Supported model catalog, validation, key masking, and stored-key
verification for the singleton ``app_settings`` row. Routes stay thin;
the catalog and provider calls live here so adding a model means editing
one map.
"""

from services.llm.base import ChatCredentials

SUPPORTED_MODELS = {
    "google": ["gemini-2.0-flash", "gemini-2.5-flash"],
    "groq": ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"],
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
        raise SettingsError(
            f"Unsupported model {model!r} for provider {provider!r}"
        )


def mask_key(api_key: str) -> str:
    """Return a display-safe mask that keeps only the last four characters."""
    if len(api_key) <= 4:
        return "••••"
    return f"••••{api_key[-4:]}"


def verify_api_key(credentials: ChatCredentials) -> tuple[bool, str | None]:
    """Run a one-token completion; return (ok, error_message)."""
    from services.llm.factory import build_chat_provider

    try:
        build_chat_provider(credentials).verify()
        return True, None
    except Exception as e:
        return False, str(e)
