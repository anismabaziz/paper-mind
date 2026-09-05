import hashlib
import uuid

import config
from services.concurrency import map_batches_concurrently
from services.hybrid import build_sparse_vector, build_sparse_vectors

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
            order the Sources panel shows, so both always agree.
    """
    shaped = {}
    for source in sorted(sources, key=lambda s: s["score"], reverse=True):
        key = source["content"].strip()
        if key and key not in shaped:
            shaped[key] = source
    return list(shaped.values())[:limit]


def matches_to_sources(matches, filename):
    """
        Turn raw index matches into the shared source shape.
    """
    sources = []
    for match in matches or []:
        metadata = (
            match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {})
        )
        content = metadata.get("content") if metadata else None
        if not content:
            continue
        score = (
            match.get("score", 0.0)
            if isinstance(match, dict)
            else (getattr(match, "score", 0.0) or 0.0)
        )
        sources.append(
            {
                "content": content,
                "document": metadata.get("pdf_name", filename),
                "chunk_index": metadata.get("chunk_index", 0),
                "page_no": metadata.get("page_no"),
                "content_hash": metadata.get("content_hash") or _content_hash(content),
                "score": float(score),
            }
        )
    return sources


class VectorService:
    # Pinecone recommends batches well under the 4 MB / 1000-vector limit.
    # Keep in sync with AIService.EMBED_BATCH_SIZE so one embedding batch maps
    # to one upsert batch without re-chunking.
    UPSERT_BATCH_SIZE = 100

    @staticmethod
    def _build_vectors_from_chunks(batch_embeddings, chunks, filename, offset):
        """Build vectors from bundled :class:`Chunk` objects (data-clump fix)."""
        vectors = []
        batch_chunks = [chunks[offset + j] for j in range(len(batch_embeddings))]
        # Chunk.text already carries the raw text for BM25
        sparse_batch = build_sparse_vectors([c.text for c in batch_chunks])
        for j, embedding in enumerate(batch_embeddings):
            chunk = batch_chunks[j]
            metadata = {
                "content": chunk.text,
                "pdf_name": filename,
                "chunk_index": chunk.chunk_index,
                "page_no": chunk.page_no,
                "content_hash": chunk.content_hash,
            }
            sparse = sparse_batch[j] if j < len(sparse_batch) else {"indices": [], "values": []}
            vectors.append(
                {
                    "id": str(uuid.uuid4()),
                    "values": embedding,
                    "sparse_values": sparse,
                    "sparse_vector": sparse,
                    "metadata": metadata,
                }
            )
        return vectors

    @staticmethod
    def _build_vectors(batch_embeddings, texts, filename, page_numbers, offset):
        # Backward-compatible shim: bundle parallel lists into Chunk, then delegate.
        # The parallel lists are a data clump; new code should call
        # `_build_vectors_from_chunks` with a `list[Chunk]`.
        from services.document_parser import Chunk

        chunks = [
            Chunk(
                text=texts[offset + j],
                page_no=page_numbers[offset + j] if page_numbers is not None and offset + j < len(page_numbers) else None,
                chunk_index=offset + j,
                content_hash=_content_hash(texts[offset + j]),
            )
            for j in range(len(batch_embeddings))
        ]
        return VectorService._build_vectors_from_chunks(batch_embeddings, chunks, filename, offset)

    @staticmethod
    def upsert_chunks(embeddings, chunks, filename):
        """Preferred entry: ``chunks`` is a ``list[Chunk]`` (bundled)."""
        if not embeddings:
            return None
        num_batches = (len(embeddings) + VectorService.UPSERT_BATCH_SIZE - 1) // VectorService.UPSERT_BATCH_SIZE
        if num_batches <= 1:
            vectors = VectorService._build_vectors_from_chunks(embeddings, chunks, filename, 0)
            return config.vector_index.upsert(vectors)
        batches: list[list[dict]] = []
        for start in range(0, len(embeddings), VectorService.UPSERT_BATCH_SIZE):
            batch_embeddings = embeddings[start : start + VectorService.UPSERT_BATCH_SIZE]
            vectors = VectorService._build_vectors_from_chunks(batch_embeddings, chunks, filename, start)
            batches.append(vectors)
        ordered_responses = map_batches_concurrently(
            batches,
            config.vector_index.upsert,
            label=f"VectorService.upsert_vectors: {len(embeddings)} vectors",
        )
        return ordered_responses[-1] if ordered_responses else None

    @staticmethod
    def upsert_vectors(embeddings, texts, filename, page_numbers=None):
        if not embeddings:
            return None
        from services.document_parser import Chunk

        # Bundle the parallel lists so the rest of the path uses Chunk
        chunks = [
            Chunk(
                text=texts[i],
                page_no=page_numbers[i] if page_numbers is not None and i < len(page_numbers) else None,
                chunk_index=i,
                content_hash=_content_hash(texts[i]),
            )
            for i in range(len(texts))
        ]
        return VectorService.upsert_chunks(embeddings, chunks, filename)

    @staticmethod
    def query_vectors(embedding, filename, top_k=FETCH_K, query_text=None, alpha=None, rerank=None):
        """
        Return shaped sources: deduped, score-ordered, bounded.

        When ``query_text`` is provided the method issues a hybrid query:
        dense embedding + BM25 sparse (``sparse_vectors`` via ``rank-bm25``)
        fused with ``RRF(k=60)`` for Qdrant or ``alpha`` blend for Pinecone.
        ``FETCH_K=50`` candidates are fetched before shaping to ``5``.

        When ``RERANK=true`` (or ``rerank=True`` explicitly) and
        ``query_text`` is present, the 50 hybrid candidates are reranked
        with a local cross-encoder (``ms-marco-MiniLM-L-6-v2`` 22M fast or
        ``bge-reranker-v2-m3`` quality) before ``shape_sources`` keeps top 5.
        Entirely local CPU, no API. ``rerank=False`` preserves legacy order
        even when the env flag is on (used by tests/evaluator).
        """
        if query_text is not None:
            sparse = build_sparse_vector(query_text)
            if not sparse["indices"]:
                sparse = None
        else:
            sparse = None

        if sparse is not None:
            # Resolve alpha: explicit arg > config > default blend
            if alpha is None:
                alpha = getattr(config, "HYBRID_ALPHA", 0.7)
            vector_backend = config._vector_backend()
            try:
                if vector_backend == "qdrant":
                    # Single Qdrant hybrid query: dense + BM25 sparse via
                    # rank-bm25 on the ``sparse`` field, fused with RRF(k=60)
                    search_results = config.vector_index.query(
                        vector=embedding,
                        top_k=top_k,
                        include_metadata=True,
                        filter={"pdf_name": filename},
                        sparse_vector=sparse,
                        sparse_values=sparse,
                    )
                else:
                    # Pinecone hybrid path: sparse_values + alpha-scaled query
                    # so both backends behave the same. Pinecone expects the
                    # dense vector scaled by ``alpha`` and sparse values scaled
                    # by ``1-alpha``.
                    scaled_dense = [v * alpha for v in embedding] if isinstance(embedding, list) else embedding
                    scaled_sparse = {
                        "indices": sparse["indices"],
                        "values": [v * (1.0 - alpha) for v in sparse["values"]],
                    }
                    search_results = config.vector_index.query(
                        vector=scaled_dense,
                        top_k=top_k,
                        include_metadata=True,
                        filter={"pdf_name": filename},
                        sparse_values=scaled_sparse,
                        sparse_vector=scaled_sparse,
                        alpha=alpha,
                    )
                    # If the Pinecone index did not fuse (e.g., fake that
                    # ignores sparse), do client-side alpha blend for parity.
                    # Detect by checking if result looks dense-only: we fetch
                    # both sides separately and blend.
                    try:
                        from services.hybrid import alpha_blend

                        # Heuristic: if search_results came from a fake that
                        # just returned dense, sparse was ignored. We can
                        # verify by issuing a dense-only query and seeing if
                        # results are identical – if so, blend manually.
                        dense_only = config.vector_index.query(
                            vector=embedding,
                            top_k=top_k,
                            include_metadata=True,
                            filter={"pdf_name": filename},
                        )
                        # Compare ids; if hybrid == dense_only, blend needed
                        hybrid_ids = [m.get("id") for m in search_results.get("matches", [])]
                        dense_ids = [m.get("id") for m in dense_only.get("matches", [])]
                        if hybrid_ids == dense_ids and hybrid_ids:
                            # Try sparse-only fetch for blending
                            try:
                                sparse_only = config.vector_index.query(
                                    vector=[0.0] * len(embedding) if isinstance(embedding, list) else embedding,
                                    top_k=top_k,
                                    include_metadata=True,
                                    filter={"pdf_name": filename},
                                    sparse_vector=sparse,
                                    sparse_values=sparse,
                                )
                                sparse_matches = sparse_only.get("matches", [])
                            except Exception:
                                sparse_matches = []
                            dense_matches = dense_only.get("matches", [])
                            blended = alpha_blend(dense_matches, sparse_matches, alpha=alpha, limit=top_k)
                            search_results = {"matches": blended}
                    except Exception:
                        pass
            except TypeError as exc:
                # Explicit warning instead of silent fallback – hybrid is
                # degraded, helps surface mis-wired fakes in tests.
                print(f"VectorService hybrid query degraded to dense (TypeError): {exc}")
                search_results = config.vector_index.query(
                    vector=embedding,
                    top_k=top_k,
                    include_metadata=True,
                    filter={"pdf_name": filename},
                )
        else:
            search_results = config.vector_index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True,
                filter={"pdf_name": filename},
            )

        matches = search_results.get("matches", []) if isinstance(search_results, dict) else getattr(search_results, "matches", [])

        sources = matches_to_sources(matches, filename)

        # Gated local reranker over FETCH_K candidates before shaping to 5.
        # Centralised in reranker.maybe_rerank so VectorService and evaluator
        # share one gate; entirely local CPU, no API.
        from services.reranker import maybe_rerank

        sources = maybe_rerank(query_text, sources, enabled=rerank)

        return shape_sources(sources)

    @staticmethod
    def delete_by_filename(filename):
        return config.vector_index.delete(filter={"pdf_name": filename})

    @staticmethod
    def delete_all():
        return config.vector_index.delete(delete_all=True)
