"""Model budgets, bounded streaming, and the answer event contract."""

import time
from types import SimpleNamespace

import pytest

from services.accounts.chat_settings_service import (
    MODEL_CATALOG,
    SUPPORTED_MODEL_DEFINITIONS,
    model_for,
)
from services.llm.base import (
    COMPLETE_FINISH_REASONS,
    RECOGNIZED_FINISH_REASONS,
    ChatBudget,
    ChatCredentials,
    EmptyAnswerError,
    LLMProvider,
    ProviderTimeoutError,
    is_transient_error,
)
from services.llm.google_provider import GoogleProvider
from services.llm.groq_provider import GroqProvider

BUDGET = ChatBudget(
    max_input_tokens=128_000,
    max_output_tokens=1_024,
    max_answer_chars=8_000,
    timeout_seconds=30.0,
)


class _Scripted(LLMProvider):
    """Run a provider whose stream is scripted, one plan per attempt."""

    name = "scripted"

    def __init__(self, attempts, budget=None, **kwargs):
        """Bind the scripted attempt list to this provider."""
        super().__init__("key", "scripted-model", budget=budget, **kwargs)
        self._attempts = list(attempts)
        self.calls = 0

    def _build_client(self):
        """No SDK client is needed for a scripted provider."""
        return None

    def verify(self):
        """Verification is not exercised here."""
        raise NotImplementedError

    def _generate_response(self, query, context, prior_turns=""):
        """One-shot generation is not exercised here."""
        raise NotImplementedError

    def _stream_response(self, query, context, prior_turns=""):
        """Yield the next scripted attempt, then raise its scripted error."""
        index = self.calls
        self.calls += 1
        tokens, error = self._attempts[index]
        yield from tokens
        if error is not None:
            raise error


def scripted(*attempts, budget=None):
    """Build a scripted provider over the given attempts."""
    return _Scripted(attempts, budget=budget)


def test_every_model_publishes_an_explicit_input_and_output_budget():
    """Each catalog entry states what it can read and what it may write."""
    for model in MODEL_CATALOG:
        assert model.max_input_tokens > 0, model.id
        assert model.answer_token_budget > 0, model.id
        assert model.answer_token_budget < model.context_window_tokens, model.id
        assert model.max_input_tokens + model.answer_token_budget <= (
            model.context_window_tokens
        ), model.id
        # A budget larger than the model's own output window is a promise the
        # provider cannot keep, and it is the kind of mismatch that only shows
        # up as a truncated answer on one model.
        assert model.answer_token_budget <= model.max_output_tokens, model.id


def test_every_model_publishes_a_generation_timeout():
    """A model with no generation bound could stream forever."""
    for model in MODEL_CATALOG:
        assert model.generation_timeout_seconds > 0, model.id


def test_every_model_publishes_a_known_finish_reason_contract():
    """Finish reasons are the vocabulary a provider reports, per model."""
    for model in MODEL_CATALOG:
        assert set(model.finish_reasons), model.id
        assert set(model.complete_finish_reasons) <= set(model.finish_reasons), model.id
        assert set(model.complete_finish_reasons), model.id
        assert set(model.finish_reasons) <= RECOGNIZED_FINISH_REASONS, model.id


def test_a_models_bounds_are_what_its_provider_runs_under():
    """The catalog's bounds are the ones the provider instance is built with."""
    from services.llm.factory import build_chat_provider

    model = model_for("groq", "openai/gpt-oss-120b")
    credentials = ChatCredentials(
        provider=model.provider,
        model=model.id,
        api_key="sk-test",
        budget=model.chat_budget(),
    )

    provider = build_chat_provider(credentials, use_cache=False)

    assert isinstance(provider.budget, ChatBudget)
    assert provider.budget == model.chat_budget()
    assert provider.budget.max_input_tokens == model.max_input_tokens
    assert provider.budget.max_output_tokens == model.answer_token_budget
    assert provider.budget.timeout_seconds == model.generation_timeout_seconds
    assert COMPLETE_FINISH_REASONS == frozenset(model.complete_finish_reasons)


def test_the_fallback_budget_fits_inside_its_own_context_window():
    """A model catalogued without a budget still gets one that holds together."""
    budget = ChatBudget()

    assert budget.max_input_tokens + budget.max_output_tokens <= 1_048_576
    assert 0 < budget.max_answer_chars
    assert budget.timeout_seconds > 0


def test_a_stream_yields_every_token_it_produced():
    """The happy path passes the provider's fragments through untouched."""
    provider = scripted((["Hello ", "there."], None))

    assert list(provider.stream_response("q", "ctx")) == ["Hello ", "there."]


def test_an_empty_stream_is_reported_rather_than_answered_with_nothing():
    """A provider that produced no text is a distinct failure, not an answer."""
    provider = scripted(([], None))

    with pytest.raises(EmptyAnswerError):
        list(provider.stream_response("q", "ctx"))


def test_whitespace_only_output_counts_as_empty():
    """A stream of blank fragments leaves the user with nothing to read."""
    provider = scripted((["   ", "\n"], None))

    with pytest.raises(EmptyAnswerError):
        list(provider.stream_response("q", "ctx"))


def test_a_transient_failure_before_any_output_is_retried_once():
    """The same provider gets one more chance while nothing is on screen."""
    provider = scripted(
        ([], TimeoutError("upstream connect reset")),
        (["Recovered."], None),
    )

    assert list(provider.stream_response("q", "ctx")) == ["Recovered."]
    assert provider.calls == 2


def test_a_retry_does_not_duplicate_tokens_the_user_already_saw():
    """Once a fragment is visible, the stream is the user's only copy of it."""
    provider = scripted((["Half "], TimeoutError("upstream reset")), (["more."], None))

    tokens = []
    with pytest.raises(TimeoutError):
        for token in provider.stream_response("q", "ctx"):
            tokens.append(token)

    assert tokens == ["Half "]
    assert provider.calls == 1


def test_a_permanent_failure_is_not_retried():
    """A rejected key will fail the same way twice; retrying only costs money."""
    provider = scripted(([], PermissionError("invalid api key")), ([], None))

    with pytest.raises(PermissionError):
        list(provider.stream_response("q", "ctx"))
    assert provider.calls == 1


def test_a_retry_stays_on_the_same_provider_and_key():
    """A retry re-runs this provider; it never bills a different account."""
    first = scripted(([], ConnectionError("reset")), (["ok"], None))
    second = scripted((["ok"], None))

    list(first.stream_response("q", "ctx"))

    assert first.calls == 2
    assert second.calls == 0


def test_an_exhausted_retry_surfaces_the_transient_failure():
    """Two transient failures in a row end the stream as a failure."""
    provider = scripted(
        ([], ConnectionError("reset")),
        ([], ConnectionError("reset")),
    )

    with pytest.raises(ConnectionError):
        list(provider.stream_response("q", "ctx"))
    assert provider.calls == 2


def test_a_provider_that_stalls_past_its_budget_times_out():
    """A stream with no finish reason stops at the model's timeout."""
    budget = ChatBudget(timeout_seconds=0.05)
    provider = _StallingProvider(budget)

    started = time.monotonic()
    with pytest.raises(ProviderTimeoutError):
        list(provider.stream_response("q", "ctx"))
    assert time.monotonic() - started < 2.0


def test_output_stops_at_the_configured_answer_budget():
    """A chat answer never grows past the budget the catalog declares."""
    budget = ChatBudget(max_output_tokens=4, max_answer_chars=8)
    provider = scripted((["a" * 10] * 3, None), budget=budget)

    answer = "".join(provider.stream_response("q", "ctx"))

    assert len(answer) == 8


def test_a_runaway_stream_is_cut_at_the_answer_budget_not_the_model_s():
    """The stored answer is bounded even when the provider ignores its cap."""
    budget = ChatBudget(
        max_output_tokens=65_536, max_answer_chars=16, timeout_seconds=30.0
    )
    provider = scripted((["x" * 64] * 4, None), budget=budget)

    assert len("".join(provider.stream_response("q", "ctx"))) == 16


def test_transience_is_read_from_the_error_not_the_provider():
    """Rate limits and dropped connections are worth another attempt."""
    assert is_transient_error(TimeoutError("read timed out"))
    assert is_transient_error(ConnectionError("connection reset by peer"))
    assert is_transient_error(RuntimeError("429 rate limit exceeded"))
    assert is_transient_error(RuntimeError("503 service unavailable"))
    assert not is_transient_error(PermissionError("invalid api key"))
    assert not is_transient_error(ValueError("bad request"))


def test_bare_status_digits_do_not_buy_a_second_bill():
    """A message that merely contains a number is not a transient failure."""
    assert not is_transient_error(RuntimeError("model gpt-500 not found"))
    assert not is_transient_error(RuntimeError("request 4291 rejected"))
    assert not is_transient_error(RuntimeError("quota 5000 tokens exceeded"))


def test_every_supported_provider_publishes_its_finish_reasons():
    """The per-provider lists are what the chat route reads as a contract."""
    for provider_models in SUPPORTED_MODEL_DEFINITIONS.values():
        for model in provider_models:
            assert model.finish_reasons
            assert model.complete_finish_reasons


class _StallingProvider(LLMProvider):
    """Stall mid-stream, the way a provider that stops responding does."""

    name = "stalling"

    def __init__(self, budget):
        """Bind the budget this stream is run under."""
        super().__init__("key", "stalling-model", budget=budget)

    def _build_client(self):
        """No SDK client is needed for a stalled stream."""
        return None

    def verify(self):
        """Verification is not exercised here."""
        raise NotImplementedError

    def _generate_response(self, query, context, prior_turns=""):
        """One-shot generation is not exercised here."""
        raise NotImplementedError

    def _stream_response(self, query, context, prior_turns=""):
        """Yield one fragment, then stall past the budget."""
        yield "thinking"
        time.sleep(30)


class _FakeGroqStream:
    """Serve Groq-shaped chunks, ending in a finish reason."""

    def __init__(self, chunks, finish_reason="stop", reason_at=None):
        """Bind the chunks, the terminal reason, and the chunk it lands on."""
        self._chunks = chunks
        self._finish_reason = finish_reason
        self._reason_at = len(chunks) - 1 if reason_at is None else reason_at

    def __iter__(self):
        """Yield each chunk; the one at ``reason_at`` carries the reason."""
        for index, text in enumerate(self._chunks):
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=text),
                        finish_reason=(
                            self._finish_reason if index == self._reason_at else None
                        ),
                    )
                ]
            )


class _FakeGroqCompletions:
    """Records the keyword arguments each completion was made with."""

    def __init__(self, stream):
        """Bind the stream a streaming call returns."""
        self._stream = stream
        self.calls = []

    def create(self, **kwargs):
        """Record the call and hand back the scripted stream."""
        self.calls.append(kwargs)
        return self._stream


class _FakeGroqClient:
    """Stand-in for the Groq SDK client."""

    def __init__(self, stream):
        """Bind the scripted stream under chat completions."""
        self.chat = SimpleNamespace(completions=_FakeGroqCompletions(stream))


def test_the_groq_call_carries_the_models_output_budget():
    """The token cap reaches the API, not just the app's own guard."""
    client = _FakeGroqClient(_FakeGroqStream(["hi"]))
    provider = GroqProvider("key", "openai/gpt-oss-120b", client=client, budget=BUDGET)

    list(provider.stream_response("q", "ctx"))

    assert client.chat.completions.calls[0]["max_tokens"] == BUDGET.max_output_tokens


def test_the_groq_stream_stops_at_its_finish_reason():
    """Chunks after a terminal finish reason are not part of the answer."""
    stream = _FakeGroqStream(["a", "b", "c"], reason_at=1)
    provider = GroqProvider(
        "key", "openai/gpt-oss-120b", client=_FakeGroqClient(stream), budget=BUDGET
    )

    assert list(provider.stream_response("q", "ctx")) == ["a", "b"]


def test_the_groq_stream_reports_the_finish_reason_it_stopped_on():
    """A length stop is distinguishable from a complete answer."""
    stream = _FakeGroqStream(["a", "b"], finish_reason="length")
    provider = GroqProvider(
        "key", "openai/gpt-oss-120b", client=_FakeGroqClient(stream), budget=BUDGET
    )

    list(provider.stream_response("q", "ctx"))

    assert provider.last_finish_reason == "length"
    assert provider.last_finish_reason not in BUDGET.complete_finish_reasons


def test_a_complete_groq_answer_reports_the_complete_finish_reason():
    """A normal stop is reported as the reason the answer is whole."""
    stream = _FakeGroqStream(["a", "b"], finish_reason="stop")
    provider = GroqProvider(
        "key", "openai/gpt-oss-120b", client=_FakeGroqClient(stream), budget=BUDGET
    )

    list(provider.stream_response("q", "ctx"))

    assert provider.last_finish_reason in BUDGET.complete_finish_reasons


class _FakeGoogleChunk:
    """Serve one Gemini-shaped streamed chunk."""

    def __init__(self, text, finish_reason=None):
        """Bind the chunk's text and its finish reason, if any."""
        self.text = text
        self.candidates = (
            [SimpleNamespace(finish_reason=finish_reason)] if finish_reason else []
        )


class _FakeGoogleModels:
    """Records the config each generation was made with."""

    def __init__(self, chunks):
        """Bind the chunks a streaming generation returns."""
        self._chunks = chunks
        self.calls = []

    def generate_content_stream(self, **kwargs):
        """Record the call and hand back the scripted chunks."""
        self.calls.append(kwargs)
        return iter(self._chunks)


class _FakeGoogleClient:
    """Stand-in for the Google GenAI client."""

    def __init__(self, chunks):
        """Bind the scripted chunks under models."""
        self.models = _FakeGoogleModels(chunks)


def test_the_google_call_carries_the_models_output_budget():
    """The token cap reaches the API, not just the app's own guard."""
    client = _FakeGoogleClient([_FakeGoogleChunk("hi", "STOP")])
    provider = GoogleProvider("key", "gemini-2.5-flash", client=client, budget=BUDGET)

    list(provider.stream_response("q", "ctx"))

    config = client.models.calls[0]["config"]
    assert config.max_output_tokens == BUDGET.max_output_tokens


def test_the_google_stream_stops_at_its_finish_reason():
    """Chunks after a terminal finish reason are not part of the answer."""
    chunks = [
        _FakeGoogleChunk("a"),
        _FakeGoogleChunk("b", "MAX_TOKENS"),
        _FakeGoogleChunk("c"),
    ]
    provider = GoogleProvider(
        "key", "gemini-2.5-flash", client=_FakeGoogleClient(chunks), budget=BUDGET
    )

    assert list(provider.stream_response("q", "ctx")) == ["a", "b"]


def test_the_google_stream_reports_a_normalized_finish_reason():
    """The Gemini vocabulary is normalized before it meets the contract."""
    chunks = [_FakeGoogleChunk("a", "MAX_TOKENS")]
    provider = GoogleProvider(
        "key", "gemini-2.5-flash", client=_FakeGoogleClient(chunks), budget=BUDGET
    )

    list(provider.stream_response("q", "ctx"))

    assert provider.last_finish_reason == "max_tokens"
    assert provider.last_finish_reason in BUDGET.finish_reasons


def test_the_prompt_is_trimmed_to_fit_the_models_input_budget():
    """A model that reads less than the app's own budgets still gets a fit prompt."""
    from services.citations import assign_source_ids
    from services.chat_context import build_chat_context, token_count
    from services.retrieval.query_expansion import expand_query

    sources = assign_source_ids(
        [{"content": "word " * 400, "chunk_index": n} for n in range(20)]
    )
    query = "What does the paper say about retrieval?"

    chat_context = build_chat_context(
        query,
        sources,
        [],
        expand_query(query, [], max_chars=2_000),
        max_turns=0,
        prior_turns_token_budget=0,
        context_token_budget=1_000_000,
        input_token_budget=1_000,
    )

    assert token_count(chat_context.context) < 1_000
    assert chat_context.dropped_sources > 0


def test_an_input_budget_no_smaller_than_the_prompt_leaves_it_alone():
    """A generous input budget does not shrink a prompt that already fits."""
    from services.citations import assign_source_ids
    from services.chat_context import build_chat_context
    from services.retrieval.query_expansion import expand_query

    sources = assign_source_ids([{"content": "short passage", "chunk_index": 0}])
    query = "What does the paper say?"

    chat_context = build_chat_context(
        query,
        sources,
        [],
        expand_query(query, [], max_chars=2_000),
        max_turns=0,
        prior_turns_token_budget=0,
        context_token_budget=6_000,
        input_token_budget=1_000_000,
    )

    assert len(chat_context.sources) == 1
    assert chat_context.dropped_sources == 0
