"""Slow tests for Qdrant: adapter contract and retrieval end to end.

Needs a reachable Qdrant (``pytest tests/slow`` skips otherwise). Runs in an
isolated collection that is removed afterwards. The end-to-end test embeds
with the real BGE-M3 weights.
"""

import pytest

pytestmark = pytest.mark.slow


@pytest.fixture()
def adapter(qdrant_client, test_collection):
    from services.retrieval.qdrant_store import QdrantIndexAdapter

    return QdrantIndexAdapter(qdrant_client, test_collection)


def _vector(value=0.1, size=1024):
    return [value] * size


def _point(filename="doc.pdf", generation=1, index=0, text="some passage text here"):
    return {
        "id": f"pt-{index}",
        "values": _vector(0.1 + index * 0.01),
        "sparse_vector": {"indices": [index + 1], "values": [1.0]},
        "metadata": {
            "content": text,
            "pdf_name": filename,
            "chunk_index": index,
            "page_no": 1,
            "content_hash": f"hash-{index}",
            "sparse_method": "hashed-tf-qdrant-idf-v1",
            "sparse_tokenizer_version": "lowercase-regex-stopwords-v1",
            "index_generation": generation,
        },
    }


class TestAdapterContract:
    def test_upsert_query_delete_roundtrip(self, adapter):
        adapter.upsert([_point()])
        matches = adapter.query(
            vector=_vector(0.1), top_k=5, filter={"pdf_name": "doc.pdf"}
        )
        assert matches["matches"]
        assert matches["matches"][0]["metadata"]["content"] == "some passage text here"
        adapter.delete(filter={"pdf_name": "doc.pdf"})
        after = adapter.query(
            vector=_vector(0.1), top_k=5, filter={"pdf_name": "doc.pdf"}
        )
        assert after["matches"] == []

    def test_generation_filter(self, adapter):
        adapter.upsert([_point(generation=1, index=0), _point(generation=2, index=1)])
        gen2 = adapter.query(
            vector=_vector(0.1),
            top_k=5,
            filter={"pdf_name": "doc.pdf", "index_generation": 2},
        )
        assert {m["metadata"]["chunk_index"] for m in gen2["matches"]} == {1}

    def test_wrong_dimension_rejected(self, adapter):
        from services.retrieval.base import VectorDimensionError

        bad = _point()
        bad["values"] = [0.1] * 16
        with pytest.raises(VectorDimensionError):
            adapter.upsert([bad])

    def test_delete_all(self, adapter):
        adapter.upsert([_point()])
        adapter.delete(delete_all=True)
        after = adapter.query(vector=_vector(0.1), top_k=5)
        assert after["matches"] == []


class TestRetrievalEndToEnd:
    @pytest.fixture()
    def service(self, qdrant_client, test_collection, cached_models):
        from services.embeddings.local_embeddings import LocalEmbeddingService
        from services.retrieval.qdrant_store import QdrantIndexAdapter
        from services.retrieval.vector_service import VectorService

        service = VectorService(
            QdrantIndexAdapter(qdrant_client, test_collection),
            reranker=None,
            embedding_service=LocalEmbeddingService("BAAI/bge-m3"),
        )
        seed = service._embeddings.embed_texts(["collection seed sentence"])
        service.upsert_vectors(seed, ["collection seed sentence"], "seed.pdf")
        service.delete_by_filename("seed.pdf")
        return service

    def test_indexed_text_is_found(self, service):
        texts = [
            "photosynthesis converts sunlight into chemical energy in leaves",
            "medieval tax policy relied on agrarian tithes and feudal dues",
        ]
        embeddings = service._embeddings.embed_texts(texts)
        service.upsert_vectors(embeddings, texts, "leaves.pdf", page_numbers=[1, 2])
        result = service.retrieve("how does photosynthesis work?", "leaves.pdf")
        assert result.outcome == "success"
        assert result.method in ("dense", "hybrid")
        assert any("photosynthesis" in source["content"] for source in result.sources)

    def test_unknown_document_is_empty(self, service):
        result = service.retrieve("photosynthesis?", "never-indexed.pdf")
        assert result.outcome == "empty"
        assert result.sources == []

    def test_delete_by_filename(self, service):
        texts = ["some indexed sentence about rivers"]
        embeddings = service._embeddings.embed_texts(texts)
        service.upsert_vectors(embeddings, texts, "rivers.pdf")
        assert service.retrieve("rivers?", "rivers.pdf").outcome == "success"
        service.delete_by_filename("rivers.pdf")
        assert service.retrieve("rivers?", "rivers.pdf").outcome == "empty"
