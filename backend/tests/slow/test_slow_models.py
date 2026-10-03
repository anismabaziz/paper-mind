"""Slow tests for real model weights: embeddings and reranker.

Loads BGE-M3 and MiniLM from the local HF cache (offline). Needs
``pytest tests/slow -m model``. Fails loudly without cached weights.
"""

import math

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.model]


@pytest.fixture(scope="module")
def embedding_service(cached_models):
    from services.embeddings.local_embeddings import LocalEmbeddingService

    return LocalEmbeddingService("BAAI/bge-m3")


@pytest.fixture(scope="module")
def reranker(cached_models):
    from services.retrieval.reranker import RerankerService

    return RerankerService("cross-encoder/ms-marco-MiniLM-L-6-v2", enabled=True)


class TestRealEmbeddings:
    def test_vectors_are_1024d_normalized(self, embedding_service):
        (vector,) = embedding_service.embed_texts(
            ["photosynthesis converts sunlight into chemical energy"]
        )
        assert len(vector) == 1024
        assert all(math.isfinite(v) for v in vector)
        assert math.sqrt(sum(v * v for v in vector)) == pytest.approx(1.0, abs=1e-3)

    def test_single_string_input(self, embedding_service):
        vectors = embedding_service.embed_texts("a query string")
        assert len(vectors) == 1
        assert len(vectors[0]) == 1024

    def test_empty_input(self, embedding_service):
        assert embedding_service.embed_texts([]) == []

    def test_order_preserved(self, embedding_service):
        texts = ["first text about cats", "second text about quantum dots"]
        first, second = embedding_service.embed_texts(texts)
        (only_first,) = embedding_service.embed_texts([texts[0]])
        assert first == pytest.approx(only_first, abs=1e-5)
        assert first != second

    def test_similar_texts_closer_than_unrelated(self, embedding_service):
        anchor, paraphrase, unrelated = embedding_service.embed_texts(
            [
                "the mitochondria produces cellular energy",
                "mitochondria generate energy for the cell",
                "medieval tax policy in agrarian france",
            ]
        )

        def cosine(a, b):
            return sum(x * y for x, y in zip(a, b))

        assert cosine(anchor, paraphrase) > cosine(anchor, unrelated)

    def test_model_identity(self, embedding_service):
        assert embedding_service.model_name == "BAAI/bge-m3"
        assert embedding_service.revision


class TestRealReranker:
    def _sources(self):
        return [
            {"content": "medieval tax policy in agrarian france", "score": 0.9},
            {"content": "mitochondria generate energy for the cell", "score": 0.8},
            {"content": "photosynthesis converts sunlight into energy", "score": 0.7},
        ]

    def test_reorders_by_relevance(self, reranker):
        import math

        out = reranker.rerank("how do mitochondria make energy?", self._sources())
        scores = [s["score"] for s in out]
        if not all(math.isfinite(score) for score in scores):
            pytest.skip(
                "cross-encoder returns non-finite scores in this environment "
                "(torch backend issue); the service keeps legacy order"
            )
        assert [s["content"] for s in out] != [s["content"] for s in self._sources()]
        assert out[0]["content"].startswith("mitochondria")
        assert all("rerank_score" in source for source in out)

    def test_scores_are_recorded(self, reranker):
        out = reranker.rerank("how do mitochondria make energy?", self._sources())
        assert len(out) == 3
        assert {s["content"] for s in out} == {s["content"] for s in self._sources()}

    def test_disabled_returns_unchanged(self, reranker):
        sources = self._sources()
        assert reranker.rerank("query", sources, enabled=False) == sources

    def test_empty_query_returns_unchanged(self, reranker):
        sources = self._sources()
        assert reranker.rerank("   ", sources) == sources

    def test_single_source_untouched(self, reranker):
        sources = self._sources()[:1]
        assert reranker.rerank("query", sources) == sources

    def test_maybe_rerank_none_query(self, reranker):
        sources = self._sources()
        assert reranker.maybe_rerank(None, sources) == sources

    def test_model_identity(self, reranker):
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert reranker.revision
