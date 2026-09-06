"""
Build a chat provider instance from a user's chat settings.

The single place that maps a provider name to its class; adding a provider
means implementing one :class:`LLMProvider` subclass and adding it here.
"""

from services.llm.base import LLMProvider
from services.llm.google_provider import GoogleProvider
from services.llm.groq_provider import GroqProvider

_PROVIDERS = {
    GoogleProvider.name: GoogleProvider,
    GroqProvider.name: GroqProvider,
}


def build_chat_provider(provider: str, model: str, api_key: str) -> LLMProvider:
    """Construct the provider a request will chat through."""
    try:
        cls = _PROVIDERS[provider]
    except KeyError:
        raise ValueError(
            f"Unsupported provider {provider!r}; expected one of {sorted(_PROVIDERS)}"
        ) from None
    return cls(api_key=api_key, model=model)
