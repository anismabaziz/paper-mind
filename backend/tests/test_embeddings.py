"""
Embedding path tests.

Embedding generation is fully local (BGE-M3 via ``sentence-transformers``):
no API key, no network. The real model is never loaded here —
``LocalEmbeddingService._embed_batch`` is stubbed so these tests cover the
batching and input-shaping contract of ``AIService.get_embeddings`` and the
normalization/truncation contract of the local service itself.
"""

import numpy as np

from services.ai_service import AIService
from services.local_embeddings import LocalEmbeddingService


class _RecordingEmbedder:
    """Stub for ``LocalEmbeddingService._embed_batch`` recording its batches."""

    def __init__(self):
        self.batches = []

    def __call__(self, texts):
        self.batches.append(list(texts))
        return [[float(len(self.batches)), 0.0] for _ in texts]


def test_get_embeddings_batches_and_preserves_order(monkeypatch):
    """100 texts split into batches of 100-free size, order preserved."""
    recorder = _RecordingEmbedder()
    monkeypatch.setattr(LocalEmbeddingService, "_embed_batch", recorder)

    texts = [f"chunk-{i}" for i in range(205)]
    result = AIService.get_embeddings(texts)

    assert recorder.batches == [
        [f"chunk-{i}" for i in range(0, 100)],
        [f"chunk-{i}" for i in range(100, 200)],
        [f"chunk-{i}" for i in range(200, 205)],
    ]
    # Every text gets exactly one vector, in input order.
    assert [v[0] for v in result] == [1.0] * 100 + [2.0] * 100 + [3.0] * 5


def test_get_embeddings_wraps_a_single_string(monkeypatch):
    """Query path passes one string; it must come back as one vector."""
    recorder = _RecordingEmbedder()
    monkeypatch.setattr(LocalEmbeddingService, "_embed_batch", recorder)

    result = AIService.get_embeddings("a single query")

    assert recorder.batches == [["a single query"]]
    assert len(result) == 1


def test_get_embeddings_empty_input_returns_empty(monkeypatch):
    """Empty list short-circuits before any batch is dispatched."""
    recorder = _RecordingEmbedder()
    monkeypatch.setattr(LocalEmbeddingService, "_embed_batch", recorder)

    assert AIService.get_embeddings([]) == []
    assert recorder.batches == []


def test_embed_batch_slices_past_1024_dims_on_old_transformers():
    """Without ``truncate_dim`` support, past-1024 dims are sliced away."""

    class _OldModel:
        def encode(self, texts, **kwargs):
            if "truncate_dim" in kwargs:
                raise TypeError("unexpected keyword argument 'truncate_dim'")
            return np.array([[3.0, 4.0, 99.0] + [0.0] * 1024])

    import services.local_embeddings as le

    original = le._model
    le._model = _OldModel()
    try:
        vectors = LocalEmbeddingService._embed_batch(["doc"])
    finally:
        le._model = original

    assert len(vectors) == 1
    assert len(vectors[0]) == 1024
    assert vectors[0][:3] == [3.0, 4.0, 99.0]
