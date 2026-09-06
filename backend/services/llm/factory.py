"""
Build a chat provider instance from a user's chat credentials.

The single place that maps a provider name to its class; adding a provider
means implementing one :class:`LLMProvider` subclass and adding it here.
"""

from services.llm.base import ChatCredentials, LLMProvider
from services.llm.google_provider import GoogleProvider
from services.llm.groq_provider import GroqProvider

_PROVIDERS = {
    GoogleProvider.name: GoogleProvider,
    GroqProvider.name: GroqProvider,
}


def build_chat_provider(credentials: ChatCredentials, client=None) -> LLMProvider:
    """
    Construct the provider a request will chat through.

    ``client`` injects a pre-built (or fake) SDK client; when omitted, the
    provider builds its real client for the stored key.
    """
    try:
        cls = _PROVIDERS[credentials.provider]
    except KeyError:
        raise ValueError(
            f"Unsupported provider {credentials.provider!r}; expected one of "
            f"{sorted(_PROVIDERS)}"
        ) from None
    return cls(api_key=credentials.api_key, model=credentials.model, client=client)
