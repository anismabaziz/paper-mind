"""
Gated local cross-encoder reranker.

Entirely local CPU, no API. Gated by ``RERANK`` (default false until hybrid
lands) — reranks the 50 hybrid candidates before ``shape_sources`` keeps top 5.

Two model options via ``RERANK_MODEL``:
  - ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (22M, ~10ms/50 docs, fast)
  - ``BAAI/bge-reranker-v2-m3`` (~80ms/50 docs, quality)

The service is intentionally lazy: ``sentence-transformers`` is only imported
when reranking is actually requested, and the model is cached after first load
(``HF_HOME`` / Docker volume ``hf_cache``). Tests monkeypatch ``rerank`` or
``_get_reranker`` so ``pytest`` never downloads weights.
"""

from __future__ import annotations

import os
import threading
import time

import config

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker = None
_reranker_lock = threading.Lock()


def _get_model_name() -> str:
    return os.getenv("RERANK_MODEL", getattr(config, "RERANK_MODEL", _DEFAULT_MODEL))


def _get_reranker():
    """
    Lazy-load the cross-encoder. Thread-safe double-checked locking.
    Raises ImportError if ``sentence-transformers`` is missing or model load fails.
    """
    global _reranker
    if _reranker is not None:
        return _reranker
    with _reranker_lock:
        if _reranker is not None:
            return _reranker
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for RERANK=true. "
                "Install it with `uv sync` or disable reranking (RERANK=false)."
            ) from exc

        model_name = _get_model_name()
        _reranker = CrossEncoder(model_name, device="cpu", trust_remote_code=True)
        return _reranker


def _is_enabled(explicit: bool | None = None) -> bool:
    if explicit is not None:
        return bool(explicit)
    try:
        return config.is_rerank_enabled()
    except Exception:
        return os.getenv("RERANK", "false").lower() in ("1", "true", "yes")


def rerank(query: str, sources: list[dict], *, enabled: bool | None = None) -> list[dict]:
    """
    Rerank ``sources`` for ``query`` using a local cross-encoder.

    - When disabled (``enabled`` is False or ``RERANK=false``), returns
      ``sources`` unchanged (legacy order preserved).
    - When enabled, scores each ``(query, content)`` pair, replaces ``score``
      with the cross-encoder relevance, and returns score-ordered results.
    - Entirely local CPU; no API call.
    - Deduping is left to ``shape_sources``; this stage only re-scores and
      re-orders. ``rerank_score`` is preserved alongside ``score`` for
      observability.
    - Latency for 50 docs is printed (cheap observability, no extra infra).
    - On model load failure, degrades to score-ordered legacy fallback with a
      warning — never raises into the request path.

    ``sources`` elements must have ``content`` and ``score`` keys (others
    preserved verbatim).
    """
    if not sources:
        return sources
    if not _is_enabled(enabled):
        return sources

    if not query or not query.strip():
        return sources

    # Small lists don't need reranking overhead; keep legacy order
    if len(sources) <= 1:
        return sources

    start = time.time()
    try:
        model = _get_reranker()
    except Exception as exc:
        print(f"Reranker degraded to legacy order (model load failed): {exc}")
        return sources

    try:
        pairs = [[query, s.get("content", "")] for s in sources]
        # CrossEncoder.predict returns numpy array or list
        scores = model.predict(pairs, batch_size=32, show_progress_bar=False, convert_to_tensor=False)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        scores = [float(v) for v in scores]

        reranked: list[dict] = []
        for src, sc in zip(sources, scores):
            reranked.append({**src, "score": sc, "rerank_score": sc})

        reranked.sort(key=lambda s: s["score"], reverse=True)
        elapsed = time.time() - start
        print(f"Reranker: reranked {len(sources)} candidates in {elapsed:.3f}s (model {_get_model_name()})")
        return reranked
    except Exception as exc:
        print(f"Reranker degraded to legacy order (scoring failed): {exc}")
        return sources


def maybe_rerank(query: str | None, sources: list[dict], enabled: bool | None = None) -> list[dict]:
    """
    Apply the gated reranker only when it should run.

    Centralises the ``query_text is not None and sources`` guard and the
    degraded-to-legacy fallback so ``VectorService`` and ``evaluation.evaluator``
    share one path instead of duplicating the try/except gate.

    Returns ``sources`` unchanged when the gate is off, the query is empty,
    or the model cannot be loaded — never raises into the request path.
    """
    if query is None or not sources:
        return sources
    if not query.strip():
        return sources
    try:
        return rerank(query, sources, enabled=enabled)
    except Exception as exc:  # pragma: no cover - defensive, rerank already handles its own errors
        print(f"Reranker skipped (maybe_rerank failed): {exc}")
        return sources


def _reset_for_tests():
    """Clear cached model so tests can re-stub without leaking state."""
    global _reranker
    with _reranker_lock:
        _reranker = None
