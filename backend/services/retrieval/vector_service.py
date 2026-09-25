"""Retrieval over a vector store: upsert, hybrid query, shaping."""

import hashlib
import uuid
from collections.abc import Callable

from services.concurrency import map_batches_concurrently
from services.retrieval.base import (
    RetrievalMethod,
    RetrievalResult,
    VectorStore,
    VectorStoreConfigurationError,
)
from services.retrieval.hybrid import (
    DEFAULT_FETCH_K,
    SPARSE_METHOD,
    TOKENIZER_VERSION,
    build_sparse_vector,
    build_sparse_vectors,
)
from services.retrieval.reranker import Reranker

FETCH_K = DEFAULT_FETCH_K
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
    embeddings,
    chunks,
    filename: str,
    offset: int = 0,
    generation: int | None = None,
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
            "sparse_method": SPARSE_METHOD,
            "sparse_tokenizer_version": TOKENIZER_VERSION,
        }
        if generation is not None:
            metadata["index_generation"] = generation
        sparse = (
            sparse_batch[j] if j < len(sparse_batch) else {"indices": [], "values": []}
        )
        identity = (
            f"papermind:{filename}:{generation}:"
            f"{chunk.content_hash}:{chunk.chunk_index}"
        )
        vectors.append(
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, identity)),
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

    def upsert_chunks(
        self,
        embeddings,
        chunks,
        filename,
        generation: int | None = None,
        check: Callable[[], None] | None = None,
    ):
        """Preferred entry: ``chunks`` is a ``list[Chunk]`` (bundled)."""
        if not embeddings:
            return None
        num_batches = (len(embeddings) + self.UPSERT_BATCH_SIZE - 1) // (
            self.UPSERT_BATCH_SIZE
        )
        if num_batches <= 1:
            if check is not None:
                check()
            vectors = build_vectors_from_chunks(
                embeddings, chunks, filename, generation=generation
            )
            return self._store.upsert(vectors)
        batches: list[list[dict]] = []
        for start in range(0, len(embeddings), self.UPSERT_BATCH_SIZE):
            batch_embeddings = embeddings[start : start + self.UPSERT_BATCH_SIZE]
            vectors = build_vectors_from_chunks(
                batch_embeddings, chunks, filename, start, generation
            )
            batches.append(vectors)
        ordered_responses = map_batches_concurrently(
            batches,
            self._store.upsert,
            label=f"VectorService.upsert_vectors: {len(embeddings)} vectors",
            before_batch=check,
        )
        return ordered_responses[-1] if ordered_responses else None

    def upsert_vectors(
        self,
        embeddings,
        texts,
        filename,
        page_numbers=None,
        generation: int | None = None,
    ):
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
        return self.upsert_chunks(embeddings, chunks, filename, generation=generation)

    def query_vectors(
        self,
        embedding,
        filename,
        top_k=FETCH_K,
        query_text=None,
        rerank=None,
        method: RetrievalMethod | None = None,
        generation: int | None = None,
        include_legacy: bool = False,
    ) -> RetrievalResult:
        """
        Return shaped sources with an explicit method and empty or success outcome.

        Queries without usable sparse terms select dense retrieval and record
        that choice in the result.
        """
        if method not in {None, "dense", "sparse", "hybrid"}:
            raise VectorStoreConfigurationError(
                f"Unsupported retrieval method: {method}"
            )

        sparse = None
        if query_text is not None and method != "dense":
            sparse = build_sparse_vector(query_text)
            if not sparse["indices"]:
                sparse = None

        if method is None:
            selected_method: RetrievalMethod = (
                "hybrid" if sparse is not None else "dense"
            )
        else:
            selected_method = method
            if selected_method != "dense" and sparse is None:
                raise VectorStoreConfigurationError(
                    f"{selected_method} retrieval requires query text"
                )

        query_filter: dict[str, object] = {"pdf_name": filename}
        if generation is not None:
            query_filter["index_generation"] = generation
        elif include_legacy:
            query_filter["index_generation"] = None
        query_args = {
            "vector": embedding,
            "top_k": top_k,
            "include_metadata": True,
            "filter": query_filter,
            "method": selected_method,
        }
        if sparse is not None:
            query_args["sparse_vector"] = sparse
        search_results = self._store.query(**query_args)

        matches = (
            search_results.get("matches", [])
            if isinstance(search_results, dict)
            else getattr(search_results, "matches", [])
        )
        sources = matches_to_sources(matches, filename)
        if self._reranker is not None:
            sources = self._reranker.maybe_rerank(query_text, sources, enabled=rerank)
        shaped = shape_sources(sources)
        actual_method = selected_method
        if isinstance(search_results, dict) and "method" in search_results:
            reported_method = search_results["method"]
            if reported_method not in {"dense", "sparse", "hybrid"}:
                raise VectorStoreConfigurationError(
                    f"Vector store reported an invalid retrieval method: {reported_method}"
                )
            actual_method = reported_method
        return RetrievalResult(
            sources=shaped,
            method=actual_method,
            outcome="success" if shaped else "empty",
        )

    def validate_generation(self, filename, generation, expected_count):
        """Verify that a generation contains the expected point count."""
        counter = getattr(self._store, "count", None)
        if not callable(counter):
            return None
        count = counter(filter={"pdf_name": filename, "index_generation": generation})
        if count != expected_count:
            raise VectorStoreConfigurationError(
                f"Index generation {generation} is incomplete: expected "
                f"{expected_count} vectors but found {count}."
            )
        return count

    def delete_unversioned(self, filename):
        """Delete vectors created before generation metadata was introduced."""
        return self._store.delete(
            filter={"pdf_name": filename, "index_generation": None}
        )

    def delete_by_generation(self, filename, generation):
        """Delete vectors from one index generation."""
        return self._store.delete(
            filter={"pdf_name": filename, "index_generation": generation}
        )

    def delete_by_filename(self, filename):
        """Delete every vector stored for one document."""
        return self._store.delete(filter={"pdf_name": filename})

    def delete_all(self):
        """Delete every vector in the store."""
        return self._store.delete(delete_all=True)
