import uuid
import config

# How many candidates the index is asked for vs. how many survive shaping.
# Asking for more than we keep gives dedupe room to work.
TOP_K = 8
MAX_RETRIEVED_SOURCES = 5


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
                "score": float(score),
            }
        )
    return sources


class VectorService:
    @staticmethod
    def upsert_vectors(embeddings, texts, filename):
        vectors = [
            {
                "id": str(uuid.uuid4()),
                "values": embedding,
                "metadata": {
                    "content": texts[i],
                    "pdf_name": filename,
                    "chunk_index": i,
                },
            }
            for i, embedding in enumerate(embeddings)
        ]
        return config.vector_index.upsert(vectors)

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
