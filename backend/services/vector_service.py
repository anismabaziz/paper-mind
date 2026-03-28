import uuid
from config import vector_index

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
        return vector_index.upsert(vectors)

    @staticmethod
    def query_vectors(embedding, filename, top_k=3):
        search_results = vector_index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            filter={"pdf_name": filename},
        )
        return [match["metadata"]["content"] for match in search_results["matches"]]

    @staticmethod
    def delete_by_filename(filename):
        return vector_index.delete(filter={"pdf_name": filename})

    @staticmethod
    def delete_all():
        return vector_index.delete(delete_all=True)
