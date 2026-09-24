"""Retrieval over a vector store: upsert, hybrid query, shaping."""

import hashlib
import uuid

from services.concurrency import map_batches_concurrently
from services.retrieval.base import VectorStore
from services.retrieval.hybrid import build_sparse_vector, build_sparse_vectors
from services.retrieval.reranker import Reranker

# How many candidates the index is asked for vs. how many survive shaping.
# Asking for more than we keep gives dedupe room to work.
TOP_K = 8
FETCH_K = 50
RRF_K = 60
MAX_RETRIEVED_SOURCES = 5


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def shape_sources(sources, limit=MAX_RETRIEVED_SOURCES):
    """
    Dedupe by content, order by score, and bound the result.

        The returned order is the order the LLM receives as context and the
    order the Sources panel shows, so both always agree. Dedupe uses raw
    ``content`` without stripping so whitespace variants are treated as
    distinct Passages.
    """
    shaped: dict[str, dict] = {}
    for source in sorted(sources, key=lambda s: s["score"], reverse=True):
        content = source.get("content", "")
        if not content or not content.strip():
            continue
        key = content
        if key not in shaped:
            shaped[key] = source
    return list(shaped.values())[:limit]


def matches_to_sources(matches, filename):
    """Turn raw index matches into the shared source shape."""
    sources = []
    for match in matches or []:
        metadata = (
            match.get("metadata", {})
            if isinstance(match, dict)
            else getattr(match, "metadata", {})
        )
        content = metadata.get("content") if metadata else None
        if not content:
            continue
        score = (
            match.get("score", 0.0)
            if isinstance(match, dict)
            else (getattr(match, "score", 0.0) or 0.0)
        )
        page_no = metadata.get("page_no")
        sources.append(
            {
                "content": content,
                "document": metadata.get("pdf_name", filename),
                "chunk_index": metadata.get("chunk_index", 0),
                "page_no": page_no,
                "page": page_no,
                "content_hash": metadata.get("content_hash") or _content_hash(content),
                "score": float(score),
            }
        )
    return sources


def build_vectors_from_chunks(
    embeddings, chunks, filename: str, offset: int = 0
) -> list[dict]:
    """Build dense and sparse index records from chunks and their embeddings."""
    vectors = []
    batch_chunks = [chunks[offset + j] for j in range(len(embeddings))]
    sparse_batch = build_sparse_vectors([chunk.text for chunk in batch_chunks])
    for j, embedding in enumerate(embeddings):
        chunk = batch_chunks[j]
        metadata = {
            "content": chunk.text,
            "pdf_name": filename,
            "chunk_index": chunk.chunk_index,
            "page_no": chunk.page_no,
            "content_hash": chunk.content_hash,
        }
        sparse = (
            sparse_batch[j] if j < len(sparse_batch) else {"indices": [], "values": []}
        )
        vectors.append(
            {
                "id": str(uuid.uuid4()),
                "values": embedding,
                "sparse_vector": sparse,
                "metadata": metadata,
            }
        )
    return vectors


class VectorService:
    """
    Indexing and retrieval over an injected :class:`VectorStore`.

    The store arrives through the constructor, as does the optional
    reranker used to re-score hybrid candidates before shaping. Batches are
    kept well under Qdrant's upsert limits, in sync with EMBED_BATCH_SIZE in
    services/embeddings/local_embeddings.py so one embedding batch maps to
    one upsert batch without re-chunking.
    """

    UPSERT_BATCH_SIZE = 100

    def __init__(self, store: VectorStore, reranker: Reranker | None = None):
        """Bind the store to index and query; the reranker is optional."""
        self._store = store
        self._reranker = reranker

    def upsert_chunks(self, embeddings, chunks, filename):
        """Preferred entry: ``chunks`` is a ``list[Chunk]`` (bundled)."""
        if not embeddings:
            return None
        num_batches = (len(embeddings) + self.UPSERT_BATCH_SIZE - 1) // (
            self.UPSERT_BATCH_SIZE
        )
        if num_batches <= 1:
            vectors = build_vectors_from_chunks(embeddings, chunks, filename)
            return self._store.upsert(vectors)
        batches: list[list[dict]] = []
        for start in range(0, len(embeddings), self.UPSERT_BATCH_SIZE):
            batch_embeddings = embeddings[start : start + self.UPSERT_BATCH_SIZE]
            vectors = build_vectors_from_chunks(
                batch_embeddings, chunks, filename, start
            )
            batches.append(vectors)
        ordered_responses = map_batches_concurrently(
            batches,
            self._store.upsert,
            label=f"VectorService.upsert_vectors: {len(embeddings)} vectors",
        )
        return ordered_responses[-1] if ordered_responses else None

    def upsert_vectors(self, embeddings, texts, filename, page_numbers=None):
        """Upsert from parallel lists, bundled into :class:`Chunk` first."""
        if not embeddings:
            return None
        from services.parsing.document_parser import Chunk

        # Bundle the parallel lists so the rest of the path uses Chunk
        chunks = [
            Chunk(
                text=texts[i],
                page_no=page_numbers[i]
                if page_numbers is not None and i < len(page_numbers)
                else None,
                chunk_index=i,
                content_hash=_content_hash(texts[i]),
            )
            for i in range(len(texts))
        ]
        return self.upsert_chunks(embeddings, chunks, filename)

    def query_vectors(
        self, embedding, filename, top_k=FETCH_K, query_text=None, rerank=None
    ):
        """
        Return shaped sources: deduped, score-ordered, bounded.

        When ``query_text`` is provided the method issues a hybrid query:
        dense embedding + BM25 sparse (``sparse_vectors`` via ``rank-bm25``)
        fused with ``RRF(k=60)`` by Qdrant.
        ``FETCH_K=50`` candidates are fetched before shaping to ``5``.

        When the injected reranker is enabled (or ``rerank=True`` explicitly)
        and ``query_text`` is present, the 50 hybrid candidates are reranked
        with a local cross-encoder (``ms-marco-MiniLM-L-6-v2`` 22M fast or
        ``bge-reranker-v2-m3`` quality) before ``shape_sources`` keeps top 5.
        Entirely local CPU, no API. ``rerank=False`` preserves legacy order
        even when the reranker is enabled (used by tests/evaluator).
        """
        sparse: dict | None = None
        if query_text is not None:
            sparse = build_sparse_vector(query_text)
            if not sparse["indices"]:
                sparse = None
        else:
            sparse = None

        if sparse is not None:
            try:
                # Single Qdrant hybrid query: dense + BM25 sparse via
                # rank-bm25 on the ``sparse`` field, fused with RRF(k=60)
                search_results = self._store.query(
                    vector=embedding,
                    top_k=top_k,
                    include_metadata=True,
                    filter={"pdf_name": filename},
                    sparse_vector=sparse,
                )
            except TypeError as exc:
                # Explicit warning instead of silent fallback – hybrid is
                # degraded, helps surface mis-wired fakes in tests.
                print(
                    f"VectorService hybrid query degraded to dense (TypeError): {exc}"
                )
                search_results = self._store.query(
                    vector=embedding,
                    top_k=top_k,
                    include_metadata=True,
                    filter={"pdf_name": filename},
                )
        else:
            search_results = self._store.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True,
                filter={"pdf_name": filename},
            )

        matches = (
            search_results.get("matches", [])
            if isinstance(search_results, dict)
            else getattr(search_results, "matches", [])
        )

        sources = matches_to_sources(matches, filename)

        # Gated local reranker over FETCH_K candidates before shaping to 5.
        # Centralised in the reranker's maybe_rerank so VectorService and
        # evaluator share one gate; entirely local CPU, no API.
        if self._reranker is not None:
            sources = self._reranker.maybe_rerank(query_text, sources, enabled=rerank)

        return shape_sources(sources)

    def delete_by_filename(self, filename):
        """Delete every vector stored for one document."""
        return self._store.delete(filter={"pdf_name": filename})

    def delete_all(self):
        """Delete every vector in the store."""
        return self._store.delete(delete_all=True)
