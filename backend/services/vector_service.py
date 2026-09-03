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
        search_results = config.vector_index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            filter={"pdf_name": filename},
        )

        matches = search_results.get("matches", []) if isinstance(search_results, dict) else getattr(search_results, "matches", [])

        contexts = []
        for match in matches or []:
            metadata = match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {})
            if metadata and metadata.get("content"):
                contexts.append(metadata["content"])

        return contexts

    @staticmethod
    def delete_by_filename(filename):
        return config.vector_index.delete(filter={"pdf_name": filename})

    @staticmethod
    def delete_all():
        return config.vector_index.delete(delete_all=True)
