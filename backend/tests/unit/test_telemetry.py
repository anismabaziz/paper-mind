"""Unit tests for telemetry redaction, cost, spans, and errors.

No exporters run here. Run with plain ``pytest``.
"""

import pytest

from services.telemetry.cost import cost_usd
from services.telemetry.redaction import (
    REDACTED,
    Redactor,
    credential_free,
    fingerprint,
)
from services.telemetry.spans import Span, Trace, Tracer, error_category

pytestmark = pytest.mark.unit


class StubModel:
    id = "test-model"
    input_cost_per_million_usd = 1.0
    output_cost_per_million_usd = 4.0


class TestCredentialFree:
    @pytest.mark.parametrize(
        "key",
        ["sk-abcdefgh12345678", "gsk_abcdefgh12345678", "AIzaabcdefghijklmnopqrstuvwx"],
    )
    def test_provider_keys_stripped(self, key):
        assert key not in credential_free(f"call with {key} inside")
        assert REDACTED in credential_free(f"call with {key} inside")

    def test_bearer_stripped(self):
        assert REDACTED in credential_free("Authorization: Bearer abcdefgh1234")

    def test_plain_text_untouched(self):
        assert credential_free("no secrets here") == "no secrets here"


class TestFingerprint:
    def test_shape(self):
        printed = fingerprint("hello")
        assert printed["redacted"] == REDACTED
        assert printed["chars"] == 5
        assert len(printed["sha256"]) == 16
        assert "hello" not in str(printed)

    def test_none_is_empty(self):
        assert fingerprint(None)["chars"] == 0

    def test_stable(self):
        assert fingerprint("same") == fingerprint("same")
        assert fingerprint("a") != fingerprint("b")


class TestRedactor:
    def test_text_attributes_fingerprinted(self):
        scrubbed = Redactor().scrub({"query": "secret question", "model": "m"})
        assert scrubbed["model"] == "m"
        assert scrubbed["query"]["redacted"] == REDACTED
        assert scrubbed["query"]["chars"] == len("secret question")

    def test_credentials_always_stripped(self):
        scrubbed = Redactor().scrub({"api_key": "sk-abcdefgh12345678"})
        assert scrubbed["api_key"] == REDACTED

    def test_credentials_stripped_even_when_capturing(self):
        redactor = Redactor(capture_text=True, capture_window_seconds=900.0)
        assert redactor.capturing is True
        scrubbed = redactor.scrub({"api_key": "sk-abcdefgh12345678", "query": "q"})
        assert scrubbed["api_key"] == REDACTED
        assert scrubbed["query"] == "q"

    def test_capture_window_expires(self):
        now = [1000.0]
        redactor = Redactor(
            capture_text=True, capture_window_seconds=10.0, clock=lambda: now[0]
        )
        assert redactor.capturing is True
        now[0] = 2000.0
        assert redactor.capturing is False
        assert redactor.scrub({"query": "q"})["query"]["redacted"] == REDACTED

    def test_no_capture_by_default(self):
        assert Redactor().capturing is False

    def test_nested_values_scrubbed(self):
        scrubbed = Redactor().scrub(
            {"sources": [{"content": "private passage text here"}]}
        )
        assert scrubbed["sources"][0]["content"]["redacted"] == REDACTED

    def test_free_text_credentials_stripped(self):
        scrubbed = Redactor().scrub({"model": "uses sk-abcdefgh12345678 today"})
        assert "sk-abcdefgh" not in scrubbed["model"]


class TestCostUsd:
    def test_arithmetic(self):
        assert cost_usd(1_000_000, 1_000_000, StubModel()) == pytest.approx(5.0)

    def test_none_when_unmeasured(self):
        assert cost_usd(None, 10, StubModel()) is None
        assert cost_usd(10, None, StubModel()) is None

    def test_zero_is_zero(self):
        assert cost_usd(0, 0, StubModel()) == 0.0


class TestErrorCategory:
    def test_typed_failures(self):
        from services.llm.base import EmptyAnswerError, ProviderTimeoutError
        from services.retrieval.base import (
            VectorDimensionError,
            VectorStoreConfigurationError,
            VectorStoreUnavailableError,
        )

        assert error_category(ProviderTimeoutError()) == "timeout"
        assert error_category(EmptyAnswerError()) == "empty_output"
        assert (
            error_category(VectorStoreUnavailableError()) == "vector_store_unavailable"
        )
        assert (
            error_category(VectorStoreConfigurationError())
            == "vector_store_configuration"
        )
        assert error_category(VectorDimensionError()) == "vector_dimension_mismatch"

    def test_unknown_is_class_name(self):
        assert error_category(KeyError("x")) == "KeyError"


class TestSpanAndTracer:
    def test_span_record_fail(self):
        span = Span(name="retrieval")
        span.record(method="dense")
        assert span.attributes["method"] == "dense"
        span.fail("timeout")
        assert span.error_category == "timeout"

    def test_null_tracer_starts_trace(self):
        trace = Tracer().start("answer.request")
        assert isinstance(trace, Trace)

    def test_trace_step_context_manager(self):
        tracer = Tracer()
        trace = tracer.start("t")
        with tracer.step(trace, "retrieval") as span:
            span.record(method="dense")
        assert len(trace.spans) == 1
        assert trace.spans[0].attributes["method"] == "dense"
        assert trace.spans[0].ended_at is not None

    def test_trace_find_and_identify(self):
        trace = Trace(name="t")
        trace.identify(document_id="d1")
        trace.span("retrieval")
        assert trace.attributes["document_id"] == "d1"
        assert trace.first("retrieval") is not None
        assert trace.first("missing") is None
