"""
Embedding path tests.

Embedding generation is fully local (BGE-M3 via ``sentence-transformers``):
no API key, no network. The real model is never loaded here — a fake
encoder is injected through the constructor, so these tests cover the
batching and input-shaping contract of ``embed_texts`` and the truncation
contract of the local service itself.
"""

import sys
import types

import numpy as np
import pytest

from services.embeddings.local_embeddings import EmbeddingService, LocalEmbeddingService
from services.models import PINNED_MODEL_REVISIONS


def _load_kwargs(service):
    """
    Run the real lazy load against a stand-in for sentence-transformers.

    Returns the keyword arguments the loader would hand the library, without
    downloading weights: the point is the arguments, not the model.
    """
    recorded = {}

    def _sentence_transformer(model_name, **kwargs):
        recorded.update(kwargs)
        return _RecordingModel()

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _sentence_transformer  # type: ignore[attr-defined]
    original = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = module
    try:
        service._get_model()
    finally:
        if original is None:
            del sys.modules["sentence_transformers"]
        else:
            sys.modules["sentence_transformers"] = original
    return recorded


class _RecordingModel:
    """Fake encoder recording the batches ``embed_texts`` dispatches."""

    def __init__(self):
        self.batches = []

    def encode(self, texts, **kwargs):
        self.batches.append(list(texts))
        return [[float(len(self.batches)), 0.0] for _ in texts]


@pytest.fixture
def service():
    """Return a service with its encoder injected through the constructor."""
    model = _RecordingModel()
    svc = LocalEmbeddingService(model_name="fake-embedding-model", model=model)
    return svc, model


def test_local_service_implements_embedding_provider(service):
    """The production adapter satisfies the provider interface used by routes."""
    svc, _ = service

    assert isinstance(svc, EmbeddingService)


def test_embed_texts_batches_and_preserves_order(service):
    """205 texts split into batches of at most 100, order preserved."""
    svc, model = service
    texts = [f"chunk-{i}" for i in range(205)]
    result = svc.embed_texts(texts)

    assert model.batches == [
        [f"chunk-{i}" for i in range(0, 100)],
        [f"chunk-{i}" for i in range(100, 200)],
        [f"chunk-{i}" for i in range(200, 205)],
    ]
    # Every text gets exactly one vector, in input order.
    assert [v[0] for v in result] == [1.0] * 100 + [2.0] * 100 + [3.0] * 5


def test_embed_texts_wraps_a_single_string(service):
    """Query path passes one string; it must come back as one vector."""
    svc, model = service
    result = svc.embed_texts("a single query")

    assert model.batches == [["a single query"]]
    assert len(result) == 1


def test_embed_texts_empty_input_returns_empty(service):
    """Empty list short-circuits before any batch is dispatched."""
    svc, model = service
    assert svc.embed_texts([]) == []
    assert model.batches == []


def test_embed_batch_slices_past_1024_dims_on_old_transformers():
    """Without ``truncate_dim`` support, past-1024 dims are sliced away."""

    class _OldModel:
        def encode(self, texts, **kwargs):
            if "truncate_dim" in kwargs:
                raise TypeError("unexpected keyword argument 'truncate_dim'")
            return np.array([[3.0, 4.0, 99.0] + [0.0] * 1024])

    svc = LocalEmbeddingService(model_name="fake-embedding-model", model=_OldModel())
    vectors = svc._embed_batch(["doc"])

    assert len(vectors) == 1
    assert len(vectors[0]) == 1024
    assert vectors[0][:3] == [3.0, 4.0, 99.0]


class TestModelLoad:
    """The weights the service asks sentence-transformers for."""

    def test_it_loads_the_revision_its_settings_name(self, monkeypatch):
        """Do test it loads the revision its settings name."""
        kwargs = _load_kwargs(LocalEmbeddingService(model_name="BAAI/bge-m3"))

        assert kwargs["revision"] == PINNED_MODEL_REVISIONS["BAAI/bge-m3"]

    def test_a_configured_revision_reaches_the_loader(self):
        """Do test a configured revision reaches the loader."""
        kwargs = _load_kwargs(
            LocalEmbeddingService(model_name="BAAI/bge-m3", revision="a" * 40)
        )

        assert kwargs["revision"] == "a" * 40

    def test_it_does_not_execute_code_from_the_model_repository(self):
        """BGE-M3 is a standard architecture; the load runs transformers code."""
        kwargs = _load_kwargs(LocalEmbeddingService(model_name="BAAI/bge-m3"))

        assert kwargs["trust_remote_code"] is False
        assert kwargs["revision"] == PINNED_MODEL_REVISIONS["BAAI/bge-m3"]

    def test_an_unpinned_model_loads_no_revision_at_all(self):
        """Do test an unpinned model loads no revision at all."""
        assert "revision" not in _load_kwargs(
            LocalEmbeddingService(model_name="acme/unknown")
        )
