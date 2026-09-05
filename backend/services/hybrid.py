"""
Hybrid dense + BM25 sparse retrieval helpers.

BM25 sparse vectors are built per-chunk from raw text, not from embeddings.
Each vector is stored alongside the dense embedding so a single hybrid query
can fuse keyword and semantic signals. Qdrant applies an ``IDF`` modifier on
the ``sparse`` field, so we only need to store raw TF weights — Qdrant does
the IDF scaling at query time.

Tokenisation mirrors the evaluator's ``_tokens`` split (lowercase, stopword
filter, length > 1) so lexical matches are stable across eval and prod.
Indices are stable hashes of the token (md5 % vocab), which keeps the sparse
space bounded without a global vocabulary table. When ``rank-bm25`` is
installed we could build corpus-level IDF, but the hash+TF path is free,
deterministic, and sufficient for the portfolio's keyword recall gate.

Fusion defaults to RRF(k=60) per the spec; alpha blend (dense * alpha +
sparse * (1-alpha)) is used for the Pinecone path which natively blends.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

try:
    from rank_bm25 import BM25Okapi  # optional, provides corpus IDF for fallback BM25

    _HAS_RANK_BM25 = True
except ImportError:  # pragma: no cover
    BM25Okapi = None  # type: ignore
    _HAS_RANK_BM25 = False

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

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Vocabulary size for hashing – large enough to keep collisions rare
VOCAB_SIZE = 30_000

RRF_K = 60
DEFAULT_FETCH_K = 50
DEFAULT_ALPHA = 0.7


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def _hash_token(token: str) -> int:
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % VOCAB_SIZE


def build_sparse_vector(text: str) -> dict:
    """
    Build a BM25-style sparse vector from ``text``.

    Returns ``{"indices": [...], "values": [...]}`` sorted by index,
    compatible with Pinecone's ``sparse_values`` shape and convertible to
    Qdrant's ``SparseVector``. TF is encoded as ``1 + log(tf)`` (BM25 TF
    saturation proxy); IDF is deferred to Qdrant's ``modifier: IDF``. When
    ``rank-bm25`` is installed the same tokenisation is used so the Python
    fallback scorer (``_sparse_dot``) can apply corpus IDF via ``BM25Okapi``.
    """
    tokens = tokenize(text)
    if not tokens:
        return {"indices": [], "values": []}

    counts = Counter(tokens)

    indices: list[int] = []
    values: list[float] = []
    for token, cnt in counts.items():
        idx = _hash_token(token)
        # Saturating TF: 1 + log(tf) keeps repeated terms from dominating.
        # IDF is supplied by Qdrant's ``modifier: IDF`` at query time; the
        # Python fallback computes IDF via ``rank-bm25`` when a corpus is
        # available (see ``qdrant_store._brute_sparse_query``).
        val = 1.0 + math.log(cnt) if cnt > 1 else 1.0
        indices.append(idx)
        values.append(float(val))

    # Sort by index for deterministic serialisation (Qdrant expects sorted)
    pairs = sorted(zip(indices, values))
    if pairs:
        indices, values = zip(*pairs)  # type: ignore[assignment]
        return {"indices": list(indices), "values": list(values)}
    return {"indices": [], "values": []}


def bm25_scores(query_tokens: list[str], corpus_tokens: list[list[str]]) -> list[float]:
    """
    Score ``corpus_tokens`` against ``query_tokens`` using ``rank-bm25``.

    Falls back to TF dot product when ``rank-bm25`` is unavailable.
    """
    if _HAS_RANK_BM25 and BM25Okapi is not None:
        bm25 = BM25Okapi(corpus_tokens)
        return bm25.get_scores(query_tokens).tolist()  # type: ignore[union-attr]
    # Fallback: TF overlap
    q_set = set(query_tokens)
    return [float(len(q_set & set(doc))) for doc in corpus_tokens]


def build_sparse_vectors(texts: list[str]) -> list[dict]:
    return [build_sparse_vector(t) for t in texts]


def to_qdrant_sparse(sparse: dict):
    """
    Convert a ``{"indices": [], "values": []}`` dict to a Qdrant
    ``SparseVector``. Falls back to a plain dict if ``qdrant_client`` is
    unavailable (tests never import it).
    """
    try:
        from qdrant_client.models import SparseVector

        return SparseVector(indices=sparse["indices"], values=sparse["values"])
    except Exception:
        return sparse


# ------------------------------------------------------------------ fusion


def rrf_fusion(
    ranked_lists: list[list[dict]],
    k: int = RRF_K,
    limit: int | None = DEFAULT_FETCH_K,
) -> list[dict]:
    """
    Reciprocal Rank Fusion over ``ranked_lists``.

    Each list is a ranking of matches (best first) with an ``id`` key.
    Scores are computed as ``sum(1 / (k + rank))`` per list, then the
    merged ranking is sorted by fused score descending. ``limit`` caps the
    returned list (default ``FETCH_K=50``). Dedupe by ``id`` keeping the
    highest fused score; ``metadata`` is carried from the first occurrence.

    This is the Qdrant-side fusion; Pinecone uses alpha blend instead.
    """
    fused: dict[str, dict] = {}
    scores: dict[str, float] = {}

    for lst in ranked_lists:
        for rank, doc in enumerate(lst, start=1):
            doc_id = doc.get("id") or doc.get("metadata", {}).get("content", "")[:64]
            # Normalise id to string for map key
            doc_id = str(doc_id)
            inc = 1.0 / (k + rank)
            scores[doc_id] = scores.get(doc_id, 0.0) + inc
            if doc_id not in fused:
                fused[doc_id] = dict(doc)

    # Attach fused score
    for doc_id, sc in scores.items():
        fused[doc_id] = {**fused[doc_id], "score": sc}

    ranked = sorted(fused.values(), key=lambda d: d["score"], reverse=True)
    if limit is not None:
        ranked = ranked[:limit]
    return ranked


def alpha_blend(
    dense_results: list[dict],
    sparse_results: list[dict],
    alpha: float = DEFAULT_ALPHA,
    limit: int | None = DEFAULT_FETCH_K,
) -> list[dict]:
    """
    Weighted score blend for Pinecone's hybrid path.

    Scores are assumed to be cosine / BM25 in [0,1] already; if sparse
    scores are unbounded they should be normalised before blending. Here
    we min-max normalise each list to [0,1] then blend.
    """
    if not dense_results and not sparse_results:
        return []

    def _norm(results: list[dict]) -> dict[str, float]:
        if not results:
            return {}
        vals = [r.get("score", 0.0) for r in results]
        lo, hi = min(vals), max(vals)
        spread = hi - lo if hi != lo else 1.0
        return {str(r.get("id", i)): (r.get("score", 0.0) - lo) / spread for i, r in enumerate(results)}

    d_norm = _norm(dense_results)
    s_norm = _norm(sparse_results)

    all_ids = set(d_norm) | set(s_norm)
    dense_by_id = {str(r.get("id", i)): r for i, r in enumerate(dense_results)}
    sparse_by_id = {str(r.get("id", i)): r for i, r in enumerate(sparse_results)}

    blended: list[dict] = []
    for doc_id in all_ids:
        d = d_norm.get(doc_id, 0.0)
        s = s_norm.get(doc_id, 0.0)
        score = alpha * d + (1.0 - alpha) * s
        # Prefer dense metadata, fallback to sparse
        base = dense_by_id.get(doc_id) or sparse_by_id.get(doc_id) or {}
        blended.append({**dict(base), "score": float(score), "id": doc_id})

    blended.sort(key=lambda d: d["score"], reverse=True)
    if limit is not None:
        blended = blended[:limit]
    return blended


def fuse_via_rrf_or_alpha(
    dense_results: list[dict],
    sparse_results: list[dict],
    method: str = "rrf",
    alpha: float = DEFAULT_ALPHA,
    k: int = RRF_K,
    limit: int | None = DEFAULT_FETCH_K,
) -> list[dict]:
    if method == "alpha":
        return alpha_blend(dense_results, sparse_results, alpha=alpha, limit=limit)
    return rrf_fusion([dense_results, sparse_results], k=k, limit=limit)
