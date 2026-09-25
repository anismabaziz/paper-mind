"""Resource limits and usage tracking for ingestion jobs."""

from __future__ import annotations

import resource
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from services.retrieval.hybrid import build_sparse_vector
from settings import (
    DEFAULT_MAX_INGESTION_MEMORY_BYTES,
    DEFAULT_MAX_INGESTION_OUTPUT_BYTES,
    DEFAULT_MAX_INGESTION_PAGES,
    DEFAULT_MAX_INGESTION_SECONDS,
    DEFAULT_MAX_INGESTION_TEXT_BYTES,
)

_LIMIT_CATEGORIES = {
    "max_pages": "page_limit_exceeded",
    "max_extracted_text_bytes": "text_limit_exceeded",
    "max_output_bytes": "output_limit_exceeded",
    "max_elapsed_seconds": "time_limit_exceeded",
    "max_memory_bytes": "memory_limit_exceeded",
}

_LIMIT_LABELS = {
    "max_pages": "page count",
    "max_extracted_text_bytes": "extracted text",
    "max_output_bytes": "index output",
    "max_elapsed_seconds": "elapsed time",
    "max_memory_bytes": "memory",
}


@dataclass(frozen=True)
class ResourceLimits:
    """Configured bounds for one ingestion run."""

    max_pages: int | None = DEFAULT_MAX_INGESTION_PAGES
    max_extracted_text_bytes: int | None = DEFAULT_MAX_INGESTION_TEXT_BYTES
    max_output_bytes: int | None = DEFAULT_MAX_INGESTION_OUTPUT_BYTES
    max_elapsed_seconds: float | None = DEFAULT_MAX_INGESTION_SECONDS
    max_memory_bytes: int | None = DEFAULT_MAX_INGESTION_MEMORY_BYTES

    @classmethod
    def from_value(cls, value: Any) -> "ResourceLimits":
        """Build limits from settings, a mapping, or a test double."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):

            def get(name: str, default: Any) -> Any:
                return value.get(name, default)
        else:

            def get(name: str, default: Any) -> Any:
                return getattr(value, name, default)

        return cls(
            max_pages=get("max_pages", DEFAULT_MAX_INGESTION_PAGES),
            max_extracted_text_bytes=get(
                "max_extracted_text_bytes", DEFAULT_MAX_INGESTION_TEXT_BYTES
            ),
            max_output_bytes=get(
                "max_output_bytes", DEFAULT_MAX_INGESTION_OUTPUT_BYTES
            ),
            max_elapsed_seconds=get(
                "max_elapsed_seconds", DEFAULT_MAX_INGESTION_SECONDS
            ),
            max_memory_bytes=get(
                "max_memory_bytes", DEFAULT_MAX_INGESTION_MEMORY_BYTES
            ),
        )

    def as_dict(self) -> dict[str, int | float]:
        """Return the configured values for the durable job record."""
        return {
            name: value
            for name, value in {
                "max_pages": self.max_pages,
                "max_extracted_text_bytes": self.max_extracted_text_bytes,
                "max_output_bytes": self.max_output_bytes,
                "max_elapsed_seconds": self.max_elapsed_seconds,
                "max_memory_bytes": self.max_memory_bytes,
            }.items()
            if value is not None
        }


class IngestionCancelled(Exception):
    """Cooperative work stopped after a cancellation request."""


class IngestionLimitExceeded(Exception):
    """A configured ingestion resource limit was exceeded."""

    def __init__(self, name: str, observed: int | float, limit: int | float) -> None:
        """Store the safe category and bounded resource values."""
        self.name = name
        self.observed = observed
        self.limit = limit
        self.category = _LIMIT_CATEGORIES[name]
        message = (
            f"{_LIMIT_LABELS[name].capitalize()} limit exceeded "
            f"({observed:g} > {limit:g}). Reduce the document or raise the "
            "corresponding ingestion limit, then retry."
        )
        super().__init__(message)
        self.message = message


def current_memory_bytes() -> int:
    """Return the process resident memory in bytes."""
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/statm", encoding="ascii") as statm:
                resident_pages = int(statm.read().split()[1])
            return resident_pages * resource.getpagesize()
        except (OSError, IndexError, ValueError):
            pass
    try:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, OSError, ValueError):
        return 0
    if sys.platform == "darwin":
        return int(value)
    return int(value * 1024)


def estimate_output_bytes(embeddings: list[Any], chunks: list[Any]) -> int:
    """Estimate bytes needed for vectors and their stored text payloads."""
    total = 0
    for index, embedding in enumerate(embeddings):
        total += 64
        try:
            total += len(embedding) * 8
        except TypeError:
            total += 8
        if index < len(chunks):
            text = getattr(chunks[index], "text", "")
            if isinstance(text, str):
                total += len(text.encode("utf-8"))
                sparse = build_sparse_vector(text)
                total += len(sparse["indices"]) * 16
    return total


class ResourceGuard:
    """Track one job's elapsed time, memory, and document resource usage."""

    def __init__(
        self,
        limits: Any,
        clock: Callable[[], float] = time.monotonic,
        memory_reader: Callable[[], int] = current_memory_bytes,
        elapsed_offset: float = 0.0,
    ) -> None:
        """Bind configured limits to one monotonic run."""
        self.limits = ResourceLimits.from_value(limits)
        self._clock = clock
        self._memory_reader = memory_reader
        self._started_at = clock()
        self._elapsed_offset = max(0.0, elapsed_offset)
        self._usage: dict[str, int | float] = {"elapsed_seconds": self._elapsed_offset}

    def observe(
        self,
        *,
        page_count: int | None = None,
        extracted_text_bytes: int | None = None,
        output_bytes: int | None = None,
    ) -> None:
        """Record a resource measurement and enforce every configured bound."""
        if page_count is not None:
            self._usage["page_count"] = page_count
        if extracted_text_bytes is not None:
            self._usage["extracted_text_bytes"] = extracted_text_bytes
        if output_bytes is not None:
            self._usage["output_bytes"] = output_bytes
        self._usage["elapsed_seconds"] = self._elapsed_offset + max(
            0.0, self._clock() - self._started_at
        )
        memory = self._memory_reader()
        if memory >= 0:
            self._usage["memory_bytes"] = memory
        self.check()

    def check(self) -> None:
        """Raise the first exceeded limit in a stable order."""
        checks = (
            ("max_pages", "page_count"),
            ("max_extracted_text_bytes", "extracted_text_bytes"),
            ("max_output_bytes", "output_bytes"),
            ("max_elapsed_seconds", "elapsed_seconds"),
            ("max_memory_bytes", "memory_bytes"),
        )
        for limit_name, usage_name in checks:
            limit = getattr(self.limits, limit_name)
            observed = self._usage.get(usage_name)
            if limit is not None and observed is not None and observed > limit:
                raise IngestionLimitExceeded(limit_name, observed, limit)

    def snapshot(self) -> dict[str, int | float]:
        """Return a copy safe to persist on the job."""
        return dict(self._usage)
