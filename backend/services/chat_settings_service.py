"""
Per-user chat settings.

Supported model catalog, validation, key masking, and stored-key
verification. Routes stay thin; the catalog and provider calls live here
so adding a model means editing one map.
"""

DEMO_EMAIL = "demo@papermind.local"

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


def verify_api_key(provider: str, model: str, api_key: str) -> tuple[bool, str | None]:
    """Run a one-token completion; return (ok, error_message)."""
    try:
        if provider == "google":
            from google import genai

            client = genai.Client(api_key=api_key)
            client.models.generate_content(
                model=model, contents="ping", config={"max_output_tokens": 1}
            )
        else:
            from groq import Groq

            Groq(api_key=api_key).chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
        return True, None
    except Exception as e:
        return False, str(e)
