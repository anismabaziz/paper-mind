"""Hashed term-frequency sparse vectors and shared fusion settings."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

# Keep in sync with ``evaluation.metrics`` / ``test_evaluation`` stopwords.
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "where",
    "which",
    "with",
}

_TOKEN_RE = re.compile(r"[a-z0-9%°]+")
# Vocabulary size for hashing – large enough to keep collisions rare
VOCAB_SIZE = 30_000

RRF_K = 60
DEFAULT_FETCH_K = 50
SPARSE_METHOD = "hashed-tf-qdrant-idf-v1"
TOKENIZER_VERSION = "lowercase-regex-stopwords-v1"


def tokenize(text: str) -> list[str]:
    """Do tokenize."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def _hash_token(token: str) -> int:
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % VOCAB_SIZE


def build_sparse_vector(text: str) -> dict:
    """Build a sorted hashed term-frequency vector for Qdrant IDF scoring."""
    tokens = tokenize(text)
    if not tokens:
        return {"indices": [], "values": []}

    counts = Counter(tokens)
    weights: dict[int, float] = {}
    for token, cnt in counts.items():
        idx = _hash_token(token)
        value = 1.0 + math.log(cnt) if cnt > 1 else 1.0
        weights[idx] = weights.get(idx, 0.0) + float(value)

    pairs = sorted(weights.items())
    indices = [idx for idx, _ in pairs]
    values = [value for _, value in pairs]
    if len(indices) != len(values) or len(indices) != len(set(indices)):
        raise ValueError("Sparse vector indices must be unique and aligned")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Sparse vector values must be finite")
    return {"indices": indices, "values": values}


def build_sparse_vectors(texts: list[str]) -> list[dict]:
    """Build one sparse vector per input text."""
    return [build_sparse_vector(t) for t in texts]
