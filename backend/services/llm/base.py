"""
The ``LLMProvider`` abstraction every chat provider implements.

An instance binds the global API key, chosen model, and that model's
``ChatBudget``, so a request never touches process-global provider state.
The budget is what makes a stream finishable: it carries the input budget
the call is allowed to read, the output budget it may write, the timeout
that stops a stalled provider, and the finish reasons the provider's API
reports when the answer is over. Subclasses implement only the SDK calls
and declare which finish reasons they understand; the base class does the
pumping, the deadline, the cap, and the one retry that is safe before the
user has seen anything.

Adding a provider means implementing one class and registering it in the
factory map.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from threading import Thread
from typing import Callable, Iterator, TypeVar, cast

#: Bound for verify round-trips so a hung provider cannot hang the route.
VERIFY_TIMEOUT_SECONDS = 15.0

#: Bound for one-shot generation; streaming carries its own per-model policy.
GENERATE_TIMEOUT_SECONDS = 60.0

#: Every finish reason the app knows how to read, normalized to lower case.
#: A provider reports its own vocabulary; the app compares normalized values so
#: one contract covers both the OpenAI-shaped and Gemini-shaped spellings.
RECOGNIZED_FINISH_REASONS = frozenset(
    {
        # OpenAI-shaped, used by Groq
        "stop",
        "length",
        "tool_calls",
        "function_call",
        "content_filter",
        # Gemini-shaped, used by Google
        "max_tokens",
        "safety",
        "recitation",
        "blocklist",
        "prohibited_content",
        "spii",
        "malformed_function_call",
        "unexpected_tool_call",
        "other",
    }
)

#: The finish reason that means the answer arrived whole. Anything else — a
#: token cap, a safety stop — is a truncated or suppressed answer and is
#: reported as such rather than stored as if it were complete.
COMPLETE_FINISH_REASONS = frozenset({"stop"})

#: How many times a stream may be attempted: the first attempt plus one retry
#: that is only ever spent before the first visible fragment.
STREAM_ATTEMPTS = 2

#: How many fragments may sit between the provider thread and the route. A
#: bounded queue caps the memory a runaway stream can claim.
FRAGMENT_QUEUE_DEPTH = 64

_T = TypeVar("_T")

#: Sentinel pushed when the provider thread is done, so an exhausted queue and
#: a stalled provider are told apart.
_END = object()


class ProviderTimeoutError(TimeoutError):
    """A stream ran past its model's timeout without reaching a finish reason."""


class EmptyAnswerError(RuntimeError):
    """A stream completed without producing any answer text."""


#: What a one-shot call answers with when the provider failed and there is
#: context to fall back on. It is retrieved document text wearing the shape of
#: an answer, so anything that grades answers has to be able to name it.
CONTEXT_FALLBACK_PREFIX = (
    "I couldn't use the language model right now, so here is relevant "
    "context from your document:"
)

#: How much fallback context a failed one-shot call is allowed to quote.
CONTEXT_FALLBACK_CHARS = 1200


def is_context_fallback(text: str | None) -> bool:
    """
    Report whether one piece of text is a failed call's fallback, not an answer.

    The fallback quotes the document back to the reader, so treating it as a
    generated answer would score the retrieval as though the model had written
    it.
    """
    return bool(text) and CONTEXT_FALLBACK_PREFIX in str(text)


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
        raise cast(BaseException, box["error"])
    return box["value"]  # type: ignore[return-value]


_TRANSIENT_MARKERS = (
    "timeout",
    "timed out",
    "deadline",
    "temporarily",
    "unavailable",
    "connection",
    "reset",
    "overloaded",
    "rate limit",
    # An SDK that reports a throttle with no body to quote leaves only its
    # exception name, which carries no space.
    "ratelimit",
    "too many requests",
    "internal server error",
    "bad gateway",
    "gateway timeout",
)


def is_transient_error(error: BaseException) -> bool:
    """
    Report whether a provider failure is worth one more attempt.

    A rejected key or a malformed request fails the same way every time, so
    retrying it only spends another call. The signal is the error's own text,
    because the two SDKs disagree on a transient base class.

    Only unambiguous phrases count. A bare status number does not: an error
    message can carry a model revision, a quota figure, or a request id that
    happens to contain those digits, and a false positive here costs the user
    a second bill for the same failure.
    """
    if isinstance(error, (PermissionError, ValueError, TypeError, KeyError)):
        return False
    text = str(error).lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def normalize_finish_reason(reason: object) -> str | None:
    """Return a finish reason in the app's normalized lower-case vocabulary."""
    if not isinstance(reason, str):
        return None
    return reason.strip().lower() or None


@dataclass(frozen=True)
class ChatBudget:
    """
    What one chat call to one model is allowed to consume.

    ``max_input_tokens`` is the model's read budget: the route trims the
    prompt until the question, the transcript, and the evidence fit inside it.
    No provider SDK takes this as a request parameter — it is a property of
    the model, so the app is the only thing that can enforce it.

    ``max_output_tokens`` is sent to the provider on every call.
    ``max_answer_chars`` is the app's own hard stop: the most answer text the
    store holds, so a provider that ignores its token cap still cannot
    produce an answer too large to persist. ``timeout_seconds`` ends a stream
    that stalls, and the finish reasons describe how the provider says "the
    answer is over".
    """

    max_input_tokens: int = 1_000_000
    max_output_tokens: int = 4_096
    max_answer_chars: int = 8_000
    timeout_seconds: float = GENERATE_TIMEOUT_SECONDS
    finish_reasons: frozenset[str] = field(
        default_factory=lambda: RECOGNIZED_FINISH_REASONS
    )
    complete_finish_reasons: frozenset[str] = field(
        default_factory=lambda: COMPLETE_FINISH_REASONS
    )


@dataclass(frozen=True, repr=False)
class ChatCredentials:
    """The global chat provider binding: provider name, model, and API key."""

    provider: str
    model: str
    api_key: str
    verification_timeout_seconds: float = VERIFY_TIMEOUT_SECONDS
    budget: ChatBudget = field(default_factory=ChatBudget)

    def __repr__(self) -> str:
        """Return a representation that never includes the API key."""
        return (
            f"ChatCredentials(provider={self.provider!r}, "
            f"model={self.model!r}, api_key='••••')"
        )


class LLMProvider(ABC):
    """A chat provider bound to the global API key, chosen model, and budget."""

    # Used in the generate-error log line; the factory keys its map on it.
    name: str = ""

    FALLBACK_ANSWER = "I don't know based on the given context."

    def __init__(
        self,
        api_key: str,
        model: str,
        client=None,
        use_cache: bool = True,
        budget: ChatBudget | None = None,
    ):
        """
        Bind the global stored key, chosen model, and its budget to this instance.

        ``client`` injects a pre-built (or fake) SDK client; when omitted,
        the real client is built for the key on first use. A provider with no
        declared budget gets the default one, so a newly catalogued model
        cannot stream unbounded by omission.
        """
        self.api_key = api_key
        self.model = model
        self.verification_timeout_seconds = VERIFY_TIMEOUT_SECONDS
        self.budget = budget or ChatBudget()
        # The reason the provider gave for ending its last stream, normalized.
        # A subclass sets it when it reads one; ``None`` means the stream ended
        # without the provider saying why.
        self.last_finish_reason: str | None = None
        # How many attempts the last stream took, counting the first one. A
        # trace reports it so a slow answer that only arrived on the retry is
        # distinguishable from one that was slow the first time.
        self.last_attempts: int = 0
        self._use_cache = use_cache
        self._client_override = client

    def _sdk_client(self):
        """Return the injected client, or the real SDK client for the key."""
        if self._client_override is not None:
            return self._client_override
        return self._build_client()

    @abstractmethod
    def _build_client(self):
        """Build the provider SDK client for the stored API key."""

    def generate_response(self, query: str, context: str, prior_turns: str = "") -> str:
        """Generate a full answer, falling back to the retrieved context."""
        try:
            return call_with_timeout(
                lambda: self._generate_response(query, context, prior_turns),
                GENERATE_TIMEOUT_SECONDS,
            )
        except Exception as e:
            print(f"AI Generation Error ({self.name}): {type(e).__name__}")
            if context and context.strip():
                return (
                    f"{CONTEXT_FALLBACK_PREFIX}\n\n{context[:CONTEXT_FALLBACK_CHARS]}"
                )
            return self.FALLBACK_ANSWER

    def stream_response(
        self, query: str, context: str, prior_turns: str = ""
    ) -> Iterator[str]:
        """
        Yield answer fragments from the bound provider, within the budget.

        ``prior_turns`` is a bounded transcript of earlier Turns in this
        Conversation, used to resolve references in the current question.

        The two ways this can end without an answer are raised rather than
        returned, so the caller can tell them apart: ``ProviderTimeoutError``
        when the model overruns its timeout, ``EmptyAnswerError`` when the
        provider completes with no text. Anything the provider raised is
        re-raised unchanged.

        A failure is retried once on this same provider, and only while nothing
        has been yielded. Once the first fragment reaches the user, a second
        attempt would either duplicate text on screen or require shipping the
        partial answer away to be deduplicated — so a failure after that point
        ends the stream. There is no cross-provider fallback either: the retry
        bills the account the user chose, and a switch would bill another one.
        """
        deadline = time.monotonic() + self.budget.timeout_seconds
        self.last_attempts = 0
        for attempt in range(1, STREAM_ATTEMPTS + 1):
            self.last_attempts = attempt
            emitted: list[str] = []
            self.last_finish_reason = None
            try:
                yield from self._bounded_stream(
                    query, context, prior_turns, deadline, emitted
                )
            except Exception as error:
                if (
                    attempt >= STREAM_ATTEMPTS
                    or emitted
                    or not is_transient_error(error)
                ):
                    raise
                continue
            if not any(fragment.strip() for fragment in emitted):
                raise EmptyAnswerError(f"{self.name} returned no answer text")
            return

    def _bounded_stream(
        self,
        query: str,
        context: str,
        prior_turns: str,
        deadline: float,
        emitted: list[str],
    ) -> Iterator[str]:
        """
        Yield fragments until the answer is whole, capped, or out of time.

        The provider runs on its own thread so a chunk that never arrives
        cannot hold the stream open past the deadline — checking the clock
        between chunks would not catch a provider that simply stopped. When
        the deadline passes, the thread is abandoned mid-flight; its remaining
        fragments go nowhere because nothing reads the queue afterwards.

        ``emitted`` collects what the caller has already been given, which is
        what decides whether a retry would duplicate text.
        """
        queue: Queue[object] = Queue(maxsize=FRAGMENT_QUEUE_DEPTH)
        self._pump(query, context, prior_turns, queue)
        written = 0
        while True:
            try:
                item = queue.get(timeout=max(deadline - time.monotonic(), 0.0))
            except Empty:
                raise ProviderTimeoutError(
                    f"{self.name} exceeded {self.budget.timeout_seconds:.1f}s"
                ) from None
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            if written >= self.budget.max_answer_chars:
                return
            fragment = str(item)
            text = fragment[: self.budget.max_answer_chars - written]
            if not text:
                continue
            written += len(text)
            emitted.append(text)
            yield text
            if written >= self.budget.max_answer_chars:
                return

    def _pump(
        self, query: str, context: str, prior_turns: str, queue: Queue[object]
    ) -> None:
        """
        Read the provider's stream onto the queue from a daemon thread.

        The queue is bounded, so a provider that outruns the route blocks here
        rather than growing the heap; the thread is abandoned when the deadline
        passes.
        """
        fragments = self._stream_response(query, context, prior_turns)
        Thread(target=self._drain, args=(fragments, queue), daemon=True).start()

    @staticmethod
    def _drain(fragments: Iterator[str], queue: Queue[object]) -> None:
        """Push one provider's fragments onto the queue, then close it out."""
        try:
            for fragment in fragments:
                try:
                    queue.put(fragment)
                except Full:
                    # The route stopped reading. Nothing left to deliver.
                    break
        except BaseException as error:  # includes GeneratorExit on teardown
            LLMProvider._offer(queue, error)
            return
        finally:
            fragments.close()
            LLMProvider._offer(queue, _END)

    @staticmethod
    def _offer(queue: Queue[object], item: object) -> None:
        """Put one item on a possibly-full queue, dropping it if it is."""
        try:
            queue.put_nowait(item)
        except Full:
            pass

    @abstractmethod
    def verify(self) -> None:
        """Raise if the API key cannot run a one-token completion."""

    def _verify_with_timeout(self, ping: Callable[[], None]) -> None:
        """Run a provider ping under the shared verify bound."""
        call_with_timeout(ping, self.verification_timeout_seconds)

    @abstractmethod
    def _generate_response(
        self, query: str, context: str, prior_turns: str = ""
    ) -> str:
        """Generate one full answer via the provider SDK."""

    @abstractmethod
    def _stream_response(
        self, query: str, context: str, prior_turns: str = ""
    ) -> Iterator[str]:
        """Yield answer fragments via the provider SDK, untruncated."""
