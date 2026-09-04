import hashlib
import uuid

import config
from services.concurrency import map_batches_concurrently

# How many candidates the index is asked for vs. how many survive shaping.
# Asking for more than we keep gives dedupe room to work.
TOP_K = 8
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
    def _build_vectors(batch_embeddings, texts, filename, page_numbers, offset):
        vectors = []
        for j, embedding in enumerate(batch_embeddings):
            idx = offset + j
            chunk = texts[idx]
            metadata = {
                "content": chunk,
                "pdf_name": filename,
                "chunk_index": idx,
                "page_no": page_numbers[idx] if page_numbers is not None and idx < len(page_numbers) else None,
                "content_hash": _content_hash(chunk),
            }
            vectors.append(
                {
                    "id": str(uuid.uuid4()),
                    "values": embedding,
                    "metadata": metadata,
                }
            )
        return vectors

    @staticmethod
    def upsert_vectors(embeddings, texts, filename, page_numbers=None):
        if not embeddings:
            return None

        num_batches = (len(embeddings) + VectorService.UPSERT_BATCH_SIZE - 1) // VectorService.UPSERT_BATCH_SIZE

        # Single batch: preserve exact prior behavior without pool overhead
        if num_batches <= 1:
            vectors = VectorService._build_vectors(embeddings, texts, filename, page_numbers, 0)
            return config.vector_index.upsert(vectors)

        batches: list[list[dict]] = []
        for start in range(0, len(embeddings), VectorService.UPSERT_BATCH_SIZE):
            batch_embeddings = embeddings[start : start + VectorService.UPSERT_BATCH_SIZE]
            vectors = VectorService._build_vectors(batch_embeddings, texts, filename, page_numbers, start)
            batches.append(vectors)

        ordered_responses = map_batches_concurrently(
            batches,
            config.vector_index.upsert,
            label=f"VectorService.upsert_vectors: {len(embeddings)} vectors",
        )
        return ordered_responses[-1] if ordered_responses else None

    @staticmethod
    def query_vectors(embedding, filename, top_k=TOP_K):
        """
            Return shaped sources: deduped, score-ordered, bounded.
        """
        search_results = config.vector_index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            filter={"pdf_name": filename},
        )

        matches = search_results.get("matches", []) if isinstance(search_results, dict) else getattr(search_results, "matches", [])

        return shape_sources(matches_to_sources(matches, filename))

    @staticmethod
    def delete_by_filename(filename):
        return config.vector_index.delete(filter={"pdf_name": filename})

    @staticmethod
    def delete_all():
        return config.vector_index.delete(delete_all=True)
