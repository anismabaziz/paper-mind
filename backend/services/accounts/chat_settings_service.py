"""
Global chat settings.

Supported model catalog, validation, key masking, and stored-key
verification for the singleton ``app_settings`` row. Routes stay thin;
the catalog and provider calls live here so adding a model means editing
one list.
"""

from dataclasses import asdict, dataclass
from typing import Literal

from services.llm.base import (
    COMPLETE_FINISH_REASONS,
    VERIFY_TIMEOUT_SECONDS,
    ChatBudget,
    ChatCredentials,
)

# How long a chat stream may run before the model is abandoned. Well above the
# time a 4096-token answer needs on a fast provider, well below the point a
# user would assume the request had died.
DEFAULT_GENERATION_TIMEOUT_SECONDS = 90.0

# The most answer text the app stores. A Turn answer is unbounded text, so this
# is a reading-budget choice rather than a column limit; it is still the last
# stop before an absurdly long answer reaches the history read model.
MAX_ANSWER_CHARS = 8_000

# How many tokens an answer may spend. A grounded answer is a few paragraphs;
# past this the model is writing past the point the citations support, so the
# budget is set where a complete grounded answer already fits.
DEFAULT_ANSWER_TOKEN_BUDGET = 1_024

# Finish reasons each provider's API reports, normalized to lower case. The
# vocabularies differ, so each provider declares its own and the app compares
# against the normalized value.
_GOOGLE_FINISH_REASONS = frozenset(
    {
        "stop",
        "max_tokens",
        "safety",
        "recitation",
        "blocklist",
        "prohibited_content",
        "spii",
        "malformed_function_call",
        "other",
    }
)
_GROQ_FINISH_REASONS = frozenset(
    {"stop", "length", "tool_calls", "function_call", "content_filter"}
)


@dataclass(frozen=True)
class ModelCapabilities:
    """
    Published capabilities and budgets for one supported chat model.

    ``context_window_tokens`` is what the model can hold; the two budgets below
    split it into what a request may read (``max_input_tokens``) and what an
    answer may write (``answer_token_budget``). The sum stays inside the window
    so a request at the input budget still has room to answer.
    """

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
    max_input_tokens: int
    answer_token_budget: int
    generation_timeout_seconds: float
    finish_reasons: frozenset[str]
    complete_finish_reasons: frozenset[str]
    max_answer_chars: int = MAX_ANSWER_CHARS

    def chat_budget(self) -> ChatBudget:
        """Return the bounds one chat call to this model runs under."""
        return ChatBudget(
            max_input_tokens=self.max_input_tokens,
            max_output_tokens=self.answer_token_budget,
            max_answer_chars=self.max_answer_chars,
            timeout_seconds=self.generation_timeout_seconds,
            finish_reasons=frozenset(self.finish_reasons),
            complete_finish_reasons=frozenset(self.complete_finish_reasons),
        )


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
        max_input_tokens=1_000_000,
        answer_token_budget=DEFAULT_ANSWER_TOKEN_BUDGET,
        generation_timeout_seconds=DEFAULT_GENERATION_TIMEOUT_SECONDS,
        finish_reasons=_GOOGLE_FINISH_REASONS,
        complete_finish_reasons=COMPLETE_FINISH_REASONS,
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
        max_input_tokens=1_000_000,
        answer_token_budget=DEFAULT_ANSWER_TOKEN_BUDGET,
        generation_timeout_seconds=DEFAULT_GENERATION_TIMEOUT_SECONDS,
        finish_reasons=_GOOGLE_FINISH_REASONS,
        complete_finish_reasons=COMPLETE_FINISH_REASONS,
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
        max_input_tokens=128_000,
        answer_token_budget=DEFAULT_ANSWER_TOKEN_BUDGET,
        generation_timeout_seconds=DEFAULT_GENERATION_TIMEOUT_SECONDS,
        finish_reasons=_GROQ_FINISH_REASONS,
        complete_finish_reasons=COMPLETE_FINISH_REASONS,
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
        max_input_tokens=128_000,
        answer_token_budget=DEFAULT_ANSWER_TOKEN_BUDGET,
        generation_timeout_seconds=DEFAULT_GENERATION_TIMEOUT_SECONDS,
        finish_reasons=_GROQ_FINISH_REASONS,
        complete_finish_reasons=COMPLETE_FINISH_REASONS,
    ),
    ModelCapabilities(
        # A third Qwen-family option alongside the two Groq-hosted open models,
        # with a quarter of their output window and a shorter one: its own
        # listing reports 16k max completion against the 131k context, so the
        # input budget has to leave room for an answer that can actually fit.
        # Prices are the ones Groq's model listing reports.
        provider="groq",
        id="qwen/qwen3.8-27b",
        context_window_tokens=131_072,
        max_output_tokens=16_384,
        structured_output=True,
        tool_use=True,
        input_cost_per_million_usd=0.80,
        output_cost_per_million_usd=4.00,
        pricing_tier="standard",
        data_location="cloud",
        timeout_seconds=VERIFY_TIMEOUT_SECONDS,
        max_input_tokens=112_000,
        answer_token_budget=DEFAULT_ANSWER_TOKEN_BUDGET,
        generation_timeout_seconds=DEFAULT_GENERATION_TIMEOUT_SECONDS,
        finish_reasons=_GROQ_FINISH_REASONS,
        complete_finish_reasons=COMPLETE_FINISH_REASONS,
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


def _published(model: ModelCapabilities) -> dict[str, object]:
    """Return one model's capabilities in a JSON-serializable shape."""
    published: dict[str, object] = asdict(model)
    # The finish-reason contract is a set; the wire format is a sorted list so
    # two identical models always publish the same bytes. Read from the
    # dataclass rather than the copy asdict made, because the copy widens the
    # element type to object and sorted() then has nothing to compare.
    for key in ("finish_reasons", "complete_finish_reasons"):
        published[key] = sorted(getattr(model, key))
    return published


def supported_models_payload() -> dict[str, list[dict[str, object]]]:
    """Return the model catalog grouped by provider."""
    return {
        provider: [_published(model) for model in models]
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
