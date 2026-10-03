"""Unit tests for hashed sparse vectors and retrieval contracts.

Pure hashing and arithmetic. Run with plain ``pytest``.
"""

import pytest

from services.retrieval.base import (
    RetrievalCandidate,
    RetrievalResult,
    VectorDimensionError,
    VectorStoreConfigurationError,
    VectorStoreError,
    VectorStoreUnavailableError,
)
from services.retrieval.hybrid import (
    DEFAULT_FETCH_K,
    RRF_K,
    SPARSE_METHOD,
    TOKENIZER_VERSION,
    VOCAB_SIZE,
    build_sparse_vector,
    build_sparse_vectors,
    tokenize,
)

pytestmark = pytest.mark.unit


class TestTokenize:
    def test_drops_stopwords_and_singles(self):
        tokens = tokenize("the cat is a triumph of science")
        assert "the" not in tokens
        assert "cat" in tokens
        assert "science" in tokens
        assert all(len(t) > 1 for t in tokens)

    def test_empty(self):
        assert tokenize("") == []

    def test_only_stopwords(self):
        assert tokenize("the and of is") == []


class TestBuildSparseVector:
    def test_empty_text(self):
        assert build_sparse_vector("") == {"indices": [], "values": []}

    def test_stopwords_only(self):
        assert build_sparse_vector("the and of") == {"indices": [], "values": []}

    def test_indices_sorted_unique_values_finite(self):
        import math

        vec = build_sparse_vector(
            "photosynthesis chlorophyll photosynthesis mitochondria"
        )
        assert vec["indices"] == sorted(vec["indices"])
        assert len(set(vec["indices"])) == len(vec["indices"])
        assert len(vec["indices"]) == len(vec["values"])
        assert all(math.isfinite(v) for v in vec["values"])
        assert all(0 <= i < VOCAB_SIZE for i in vec["indices"])

    def test_repeated_term_weighs_more(self):
        once = build_sparse_vector("photosynthesis mitochondria")
        twice = build_sparse_vector("photosynthesis photosynthesis mitochondria")
        by_index_once = dict(zip(once["indices"], once["values"]))
        by_index_twice = dict(zip(twice["indices"], twice["values"]))
        shared = set(by_index_once) & set(by_index_twice)
        assert shared
        assert sum(by_index_twice[i] for i in shared) > sum(
            by_index_once[i] for i in shared
        )

    def test_deterministic(self):
        text = "retrieval augmented generation over research papers"
        assert build_sparse_vector(text) == build_sparse_vector(text)

    def test_batch_matches_single(self):
        texts = ["first document text", "", "second text here"]
        assert build_sparse_vectors(texts) == [build_sparse_vector(t) for t in texts]

    def test_constants_sane(self):
        assert VOCAB_SIZE == 30_000
        assert RRF_K == 60
        assert DEFAULT_FETCH_K == 50
        assert SPARSE_METHOD
        assert TOKENIZER_VERSION


class TestRetrievalContracts:
    def test_candidate_to_dict(self):
        candidate = RetrievalCandidate(
            rank=1, score=0.9, content_hash="h", chunk_index=3
        )
        payload = candidate.to_dict()
        assert payload["rank"] == 1
        assert payload["selected"] is False
        assert payload["fused_rank"] is None

    def test_result_defaults(self):
        result = RetrievalResult(sources=[], method="dense", outcome="empty")
        assert result.candidates == ()
        assert result.rerank is None

    def test_error_hierarchy(self):
        assert issubclass(VectorStoreUnavailableError, VectorStoreError)
        assert issubclass(VectorStoreConfigurationError, VectorStoreError)
        assert issubclass(VectorDimensionError, VectorStoreError)
        assert issubclass(VectorDimensionError, ValueError)

    def test_dimension_error_with_sizes(self):
        error = VectorDimensionError(expected=1024, got=512)
        assert error.expected == 1024
        assert error.got == 512
        assert "1024" in str(error) and "512" in str(error)

    def test_dimension_error_without_sizes(self):
        error = VectorDimensionError()
        assert "dimension mismatch" in str(error).lower()
