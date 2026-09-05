"""
Shared helper for concurrent batch execution.

Both ``AIService.get_embeddings`` and ``VectorService.upsert_vectors``
split work into 100-item batches and run them with a 4-worker thread pool.
The helper centralises the pool handling, ordering, timing, and the
narrow fallback that only applies to thread-pool infrastructure failures
— business errors (retry-exhausted embeddings, upsert rejections) are
propagated rather than masked.
"""

import concurrent.futures
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def map_batches_concurrently(
    batches: list[T],
    func: Callable[[T], R],
    *,
    label: str,
    max_workers: int = 4,
) -> list[R]:
    """
    Run ``func`` over ``batches`` concurrently, preserving input order.

    A single batch bypasses the pool to avoid threading overhead. For
    multiple batches a ``ThreadPoolExecutor`` with ``max_workers`` is
    used; results are collected in submission order. If the pool itself
    cannot be created or a task cannot be submitted (infrastructure
    failure), the call falls back to a sequential loop - the fallback
    reuses the same ``func`` so retry/backoff semantics stay per batch.
    Business errors raised by ``func`` (via ``future.result()``) are
    not caught here and propagate to the caller.
    """
    if not batches:
        return []

    if len(batches) == 1:
        return [func(batches[0])]

    start = time.time()

    # Infrastructure phase: pool creation and submission. Any failure here
    # is a thread-pool problem and warrants a sequential fallback. Business
    # errors from ``func`` are not raised until ``future.result()`` below
    # and must not be masked.
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(func, batch): idx for idx, batch in enumerate(batches)
            }
    except Exception as exc:
        elapsed = time.time() - start
        print(
            f"{label}: concurrent batching failed after {elapsed:.2f}s "
            f"({exc}), falling back to sequential"
        )
        return [func(batch) for batch in batches]

    # Business phase: collection. Let func errors propagate so a
    # retry-exhausted embedding or an upsert rejection is not hidden.
    ordered: list[R | None] = [None] * len(batches)
    for future in concurrent.futures.as_completed(futures):
        idx = futures[future]
        ordered[idx] = future.result()

    elapsed = time.time() - start
    print(
        f"{label} in {len(batches)} batches concurrently in {elapsed:.2f}s ({max_workers} workers)"
    )
    return ordered  # type: ignore[return-value]
