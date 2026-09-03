import uuid
import config

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
    def query_vectors(embedding, filename, top_k=3):
        """Return retrieved chunks as source dicts the repository can persist."""
        search_results = config.vector_index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            filter={"pdf_name": filename},
        )

        matches = search_results.get("matches", []) if isinstance(search_results, dict) else getattr(search_results, "matches", [])

        sources = []
        for match in matches or []:
            metadata = match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {})
            content = metadata.get("content") if metadata else None
            if not content:
                continue
            score = match.get("score", 0.0) if isinstance(match, dict) else (getattr(match, "score", 0.0) or 0.0)
            sources.append(
                {
                    "content": content,
                    "document": metadata.get("pdf_name", filename),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "score": float(score),
                }
            )

        return sources

    @staticmethod
    def delete_by_filename(filename):
        return config.vector_index.delete(filter={"pdf_name": filename})

    @staticmethod
    def delete_all():
        return config.vector_index.delete(delete_all=True)
