"""Unit tests for ingestion resource limits.

Covers ``ResourceLimits``, ``IngestionLimitExceeded``, ``estimate_output_bytes``,
and ``ResourceGuard``. Timing and memory are faked with injected callables so
these stay instant. Real documents and real memory pressure belong in
tests/slow.
"""

import pytest

from services.ingestion.limits import (
    IngestionLimitExceeded,
    ResourceGuard,
    ResourceLimits,
    current_memory_bytes,
    estimate_output_bytes,
)
from settings import (
    DEFAULT_MAX_INGESTION_MEMORY_BYTES,
    DEFAULT_MAX_INGESTION_OUTPUT_BYTES,
    DEFAULT_MAX_INGESTION_PAGES,
    DEFAULT_MAX_INGESTION_SECONDS,
    DEFAULT_MAX_INGESTION_TEXT_BYTES,
)

pytestmark = pytest.mark.unit


class FakeClock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


class FakeChunk:
    def __init__(self, text):
        self.text = text


class TestResourceLimits:
    def test_defaults_match_settings(self):
        limits = ResourceLimits()
        assert limits.max_pages == DEFAULT_MAX_INGESTION_PAGES
        assert limits.max_extracted_text_bytes == DEFAULT_MAX_INGESTION_TEXT_BYTES
        assert limits.max_output_bytes == DEFAULT_MAX_INGESTION_OUTPUT_BYTES
        assert limits.max_elapsed_seconds == DEFAULT_MAX_INGESTION_SECONDS
        assert limits.max_memory_bytes == DEFAULT_MAX_INGESTION_MEMORY_BYTES

    def test_from_none_returns_defaults(self):
        assert ResourceLimits.from_value(None) == ResourceLimits()

    def test_from_instance_passes_through(self):
        limits = ResourceLimits(max_pages=5)
        assert ResourceLimits.from_value(limits) is limits

    def test_from_mapping_full(self):
        limits = ResourceLimits.from_value(
            {
                "max_pages": 1,
                "max_extracted_text_bytes": 2,
                "max_output_bytes": 3,
                "max_elapsed_seconds": 4.0,
                "max_memory_bytes": 5,
            }
        )
        assert (limits.max_pages, limits.max_extracted_text_bytes) == (1, 2)
        assert (limits.max_output_bytes, limits.max_elapsed_seconds) == (3, 4.0)
        assert limits.max_memory_bytes == 5

    def test_from_mapping_partial_uses_defaults(self):
        limits = ResourceLimits.from_value({"max_pages": 10})
        assert limits.max_pages == 10
        assert limits.max_extracted_text_bytes == DEFAULT_MAX_INGESTION_TEXT_BYTES

    def test_from_object_with_attributes(self):
        class Config:
            max_pages = 7

        limits = ResourceLimits.from_value(Config())
        assert limits.max_pages == 7
        assert limits.max_output_bytes == DEFAULT_MAX_INGESTION_OUTPUT_BYTES

    def test_as_dict_skips_none(self):
        limits = ResourceLimits(max_pages=None, max_output_bytes=10)
        payload = limits.as_dict()
        assert "max_pages" not in payload
        assert payload["max_output_bytes"] == 10

    def test_as_dict_full(self):
        payload = ResourceLimits(max_pages=1, max_elapsed_seconds=2.5).as_dict()
        assert payload["max_pages"] == 1
        assert payload["max_elapsed_seconds"] == 2.5


class TestIngestionLimitExceeded:
    @pytest.mark.parametrize(
        ("name", "category", "label"),
        [
            ("max_pages", "page_limit_exceeded", "Page count"),
            ("max_extracted_text_bytes", "text_limit_exceeded", "Extracted text"),
            ("max_output_bytes", "output_limit_exceeded", "Index output"),
            ("max_elapsed_seconds", "time_limit_exceeded", "Elapsed time"),
            ("max_memory_bytes", "memory_limit_exceeded", "Memory"),
        ],
    )
    def test_category_and_message(self, name, category, label):
        error = IngestionLimitExceeded(name, 11, 10)
        assert error.category == category
        assert error.name == name
        assert error.observed == 11
        assert error.limit == 10
        assert label in error.message
        assert isinstance(error, Exception)

    def test_message_suggests_retry(self):
        error = IngestionLimitExceeded("max_pages", 5, 4)
        assert "retry" in error.message.lower()


class TestEstimateOutputBytes:
    def test_empty_is_zero(self):
        assert estimate_output_bytes([], []) == 0

    def test_counts_embedding_floats(self):
        assert estimate_output_bytes([[0.1] * 10], []) == 64 + 10 * 8

    def test_non_sized_embedding_counts_fallback(self):
        assert estimate_output_bytes([None], []) == 64 + 8

    def test_chunk_text_added_once_per_embedding(self):
        chunks = [FakeChunk("hello")]
        without = estimate_output_bytes([[0.0]], [])
        with_text = estimate_output_bytes([[0.0]], chunks)
        assert with_text > without
        assert with_text >= without + len("hello".encode("utf-8"))

    def test_extra_chunks_ignored(self):
        one = estimate_output_bytes([[0.0]], [FakeChunk("a"), FakeChunk("b")])
        assert one == estimate_output_bytes([[0.0]], [FakeChunk("a")])

    def test_non_string_text_ignored(self):
        chunks = [FakeChunk(None)]
        assert estimate_output_bytes([[0.0]], chunks) == 64 + 8


class TestCurrentMemoryBytes:
    def test_returns_non_negative_int(self):
        assert isinstance(current_memory_bytes(), int)
        assert current_memory_bytes() >= 0


class TestResourceGuard:
    def _guard(self, clock=None, memory=0, **limits):
        return ResourceGuard(
            ResourceLimits(**limits),
            clock=clock or FakeClock(),
            memory_reader=lambda: memory,
        )

    def test_observe_records_usage(self):
        guard = self._guard()
        guard.observe(page_count=3, extracted_text_bytes=10, output_bytes=20)
        snapshot = guard.snapshot()
        assert snapshot["page_count"] == 3
        assert snapshot["extracted_text_bytes"] == 10
        assert snapshot["output_bytes"] == 20
        assert "elapsed_seconds" in snapshot

    def test_snapshot_returns_copy(self):
        guard = self._guard()
        guard.observe(page_count=1)
        snapshot = guard.snapshot()
        snapshot["page_count"] = 999
        assert guard.snapshot()["page_count"] == 1

    def test_page_limit(self):
        guard = self._guard(max_pages=2)
        with pytest.raises(IngestionLimitExceeded) as exc_info:
            guard.observe(page_count=3)
        assert exc_info.value.category == "page_limit_exceeded"

    def test_text_limit(self):
        guard = self._guard(max_extracted_text_bytes=5)
        with pytest.raises(IngestionLimitExceeded) as exc_info:
            guard.observe(extracted_text_bytes=6)
        assert exc_info.value.category == "text_limit_exceeded"

    def test_output_limit(self):
        guard = self._guard(max_output_bytes=5)
        with pytest.raises(IngestionLimitExceeded) as exc_info:
            guard.observe(output_bytes=6)
        assert exc_info.value.category == "output_limit_exceeded"

    def test_time_limit_with_fake_clock(self):
        clock = FakeClock(now=100.0)
        guard = self._guard(clock=clock, max_elapsed_seconds=10.0)
        guard.observe()
        clock.now = 115.0
        with pytest.raises(IngestionLimitExceeded) as exc_info:
            guard.observe()
        assert exc_info.value.category == "time_limit_exceeded"

    def test_memory_limit(self):
        guard = self._guard(memory=10**9, max_memory_bytes=100)
        with pytest.raises(IngestionLimitExceeded) as exc_info:
            guard.observe()
        assert exc_info.value.category == "memory_limit_exceeded"

    def test_equal_to_limit_passes(self):
        guard = self._guard(max_pages=2)
        guard.observe(page_count=2)

    def test_check_order_pages_before_memory(self):
        guard = self._guard(memory=10**12, max_pages=1, max_memory_bytes=1)
        guard._usage["page_count"] = 5
        guard._usage["memory_bytes"] = 10**12
        with pytest.raises(IngestionLimitExceeded) as exc_info:
            guard.check()
        assert exc_info.value.name == "max_pages"

    def test_elapsed_offset_counts_toward_limit(self):
        clock = FakeClock(now=100.0)
        guard = ResourceGuard(
            ResourceLimits(max_elapsed_seconds=10.0),
            clock=clock,
            memory_reader=lambda: 0,
            elapsed_offset=11.0,
        )
        with pytest.raises(IngestionLimitExceeded):
            guard.observe()

    def test_negative_clock_delta_clamped(self):
        clock = FakeClock(now=100.0)
        guard = self._guard(clock=clock, max_elapsed_seconds=1000.0)
        clock.now = 90.0
        guard.observe()
        assert guard.snapshot()["elapsed_seconds"] >= 0

    def test_negative_memory_not_recorded(self):
        guard = ResourceGuard(
            ResourceLimits(),
            clock=FakeClock(),
            memory_reader=lambda: -1,
        )
        guard.observe()
        assert "memory_bytes" not in guard.snapshot()

    def test_none_limit_never_raises(self):
        guard = self._guard(
            max_pages=None,
            max_extracted_text_bytes=None,
            max_output_bytes=None,
            max_elapsed_seconds=None,
            max_memory_bytes=None,
        )
        guard.observe(page_count=10**9, extracted_text_bytes=10**9, output_bytes=10**9)
