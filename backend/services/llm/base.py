"""
The ``LLMProvider`` abstraction every chat provider implements.

An instance binds one user's API key and chosen model, so a request never
touches process-global provider state. The base class carries the shared
generate/stream plumbing — the generate fallback to the retrieved context,
and the streaming contract where failures propagate to the caller so the
route can surface them as an SSE error event. Subclasses implement only the
SDK calls. Adding a provider means implementing one class and registering it
in the factory map.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class ChatCredentials:
    """One user's chat provider binding: provider name, model, and API key."""

    provider: str
    model: str
    api_key: str


class LLMProvider(ABC):
    """A chat provider bound to one user's API key and chosen model."""

    # Used in the generate-error log line; the factory keys its map on it.
    name: str = ""

    FALLBACK_ANSWER = "I don't know based on the given context."

    def __init__(self, api_key: str, model: str, client=None):
        """
        Bind the user's stored key and chosen model to this instance.

        ``client`` injects a pre-built (or fake) SDK client; when omitted,
        the real client is built for the key on first use.
        """
        self.api_key = api_key
        self.model = model
        self._client_override = client

    def _sdk_client(self):
        """Return the injected client, or the real SDK client for the key."""
        if self._client_override is not None:
            return self._client_override
        return self._build_client()

    @abstractmethod
    def _build_client(self):
        """Build the provider SDK client for the stored API key."""

    def generate_response(self, query: str, context: str) -> str:
        """Generate a full answer, falling back to the retrieved context."""
        try:
            return self._generate_response(query, context)
        except Exception as e:
            print(f"AI Generation Error ({self.name}): {e}")
            if context and context.strip():
                return (
                    "I couldn't use the language model right now, so here is relevant context from your document:\n\n"
                    f"{context[:1200]}"
                )
            return self.FALLBACK_ANSWER

    def stream_response(self, query: str, context: str) -> Iterator[str]:
        """
        Yield answer fragments from the bound provider.

        Failures propagate: the caller surfaces them as an SSE error event.
        There is no cross-provider fallback — the user picked this provider.
        """
        yield from self._stream_response(query, context)

    @abstractmethod
    def verify(self) -> None:
        """Raise if the API key cannot run a one-token completion."""

    @abstractmethod
    def _generate_response(self, query: str, context: str) -> str:
        """Generate one full answer via the provider SDK."""

    @abstractmethod
    def _stream_response(self, query: str, context: str) -> Iterator[str]:
        """Yield answer fragments via the provider SDK."""
