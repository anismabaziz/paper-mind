"""
Embedding path tests.

Embedding generation is fully local (BGE-M3 via ``sentence-transformers``):
no API key, no network. The real model is never loaded here —
``LocalEmbeddingService._embed_batch`` is stubbed so these tests cover the
batching and input-shaping contract of ``embed_texts`` and the truncation
contract of the local service itself.
"""

import numpy as np
import pytest

from services.embeddings.local_embeddings import LocalEmbeddingService


class _RecordingEmbedder:
    """Stub for ``LocalEmbeddingService._embed_batch`` recording its batches."""

    def __init__(self):
        self.batches = []

    def __call__(self, texts):
        self.batches.append(list(texts))
        return [[float(len(self.batches)), 0.0] for _ in texts]


@pytest.fixture
def service(monkeypatch):
    """Return a service with its per-batch seam replaced by a recording stub."""
    svc = LocalEmbeddingService(model_name="fake-embedding-model")
    stub = _RecordingEmbedder()
    monkeypatch.setattr(svc, "_embed_batch", stub)
    return svc, stub


def test_embed_texts_batches_and_preserves_order(service):
    """205 texts split into batches of at most 100, order preserved."""
    svc, recorder = service
    texts = [f"chunk-{i}" for i in range(205)]
    result = svc.embed_texts(texts)

    assert recorder.batches == [
        [f"chunk-{i}" for i in range(0, 100)],
        [f"chunk-{i}" for i in range(100, 200)],
        [f"chunk-{i}" for i in range(200, 205)],
    ]
    # Every text gets exactly one vector, in input order.
    assert [v[0] for v in result] == [1.0] * 100 + [2.0] * 100 + [3.0] * 5


def test_embed_texts_wraps_a_single_string(service):
    """Query path passes one string; it must come back as one vector."""
    svc, recorder = service
    result = svc.embed_texts("a single query")

    assert recorder.batches == [["a single query"]]
    assert len(result) == 1


def test_embed_texts_empty_input_returns_empty(service):
    """Empty list short-circuits before any batch is dispatched."""
    svc, recorder = service
    assert svc.embed_texts([]) == []
    assert recorder.batches == []


def test_embed_batch_slices_past_1024_dims_on_old_transformers():
    """Without ``truncate_dim`` support, past-1024 dims are sliced away."""

    class _OldModel:
        def encode(self, texts, **kwargs):
            if "truncate_dim" in kwargs:
                raise TypeError("unexpected keyword argument 'truncate_dim'")
            return np.array([[3.0, 4.0, 99.0] + [0.0] * 1024])

    svc = LocalEmbeddingService(model_name="fake-embedding-model")
    svc._model = _OldModel()
    vectors = svc._embed_batch(["doc"])

    assert len(vectors) == 1
    assert len(vectors[0]) == 1024
    assert vectors[0][:3] == [3.0, 4.0, 99.0]
