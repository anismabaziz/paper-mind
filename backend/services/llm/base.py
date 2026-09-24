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
from threading import Thread
from typing import Callable, Iterator, TypeVar

#: Bound for verify round-trips so a hung provider cannot hang the route.
VERIFY_TIMEOUT_SECONDS = 15.0

#: Bound for one-shot generation; streaming stays unbounded by design.
GENERATE_TIMEOUT_SECONDS = 60.0

_T = TypeVar("_T")


def call_with_timeout(fn: Callable[[], _T], timeout: float) -> _T:
    """Run ``fn`` with a bound, raising TimeoutError when it overruns."""
    box: dict[str, _T | BaseException] = {}

    def _run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # propagate, including Cancelled
            box["error"] = exc

    # Daemon so a hung provider call never blocks process exit; the
    # abandoned call keeps running in the background until it returns.
    worker = Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(f"provider call exceeded {timeout:.1f}s")
    if "error" in box:
        raise box["error"]
    return box["value"]  # type: ignore[return-value]


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
            return call_with_timeout(
                lambda: self._generate_response(query, context),
                GENERATE_TIMEOUT_SECONDS,
            )
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

    def _verify_with_timeout(self, ping: Callable[[], None]) -> None:
        """Run a provider ping under the shared verify bound."""
        call_with_timeout(ping, VERIFY_TIMEOUT_SECONDS)

    @abstractmethod
    def _generate_response(self, query: str, context: str) -> str:
        """Generate one full answer via the provider SDK."""

    @abstractmethod
    def _stream_response(self, query: str, context: str) -> Iterator[str]:
        """Yield answer fragments via the provider SDK."""
