"""Slow end-to-end ingestion: parse, embed, index, activate, retrieve.

Real Postgres, real Qdrant test collection, real BGE-M3 weights, real
pymupdf parsing. Needs ``pytest tests/slow`` with the compose stack up.
"""

import pytest

from conftest import TEST_COLLECTION, make_pdf_bytes

pytestmark = pytest.mark.slow

FILENAME = "slow-e2e.pdf"
PAGES = [
    "Photosynthesis converts sunlight into chemical energy inside chloroplasts.",
    "The Calvin cycle fixes carbon dioxide into sugars in the stroma.",
]


@pytest.fixture()
def stack(
    repositories, slow_settings, qdrant_client, test_collection, cached_models, tmp_path
):
    from services.embeddings.local_embeddings import LocalEmbeddingService
    from services.indexing.manifest import manifest_builder
    from services.ingestion.worker import IngestionWorker
    from services.parsing.document_parser import DocumentIngestor
    from services.retrieval.qdrant_store import QdrantIndexAdapter
    from services.retrieval.vector_service import VectorService
    from storage import LocalStorage

    storage = LocalStorage(tmp_path / "worker-storage")
    embeddings = LocalEmbeddingService("BAAI/bge-m3")
    vectors = VectorService(
        QdrantIndexAdapter(qdrant_client, TEST_COLLECTION),
        reranker=None,
        embedding_service=embeddings,
    )
    worker = IngestionWorker(
        repositories,
        storage,
        DocumentIngestor("false", 512, 50),
        embeddings,
        vectors,
        worker_id="slow-test-worker",
        manifest_builder=manifest_builder(slow_settings),
    )
    return {
        "repositories": repositories,
        "settings": slow_settings,
        "storage": storage,
        "vectors": vectors,
        "worker": worker,
    }


class TestWorkerDrain:
    def test_full_ingestion(self, stack):
        repositories = stack["repositories"]
        record, job = repositories.files.create_file_with_job(
            FILENAME, title="Slow Paper"
        )
        stack["storage"].save(FILENAME, make_pdf_bytes(PAGES))

        assert stack["worker"].drain() == 1

        finished = repositories.ingestion_jobs.get(job["id"])
        assert finished["state"] == "ready"
        stored = repositories.files.get_file(FILENAME)
        assert stored["is_processed"] is True
        assert stored["index_generation"] == 1
        assert stored["index_manifest"]
        assert repositories.conversations.get_conversation_id(record["id"])

        result = stack["vectors"].retrieve(
            "how does photosynthesis fix carbon?",
            FILENAME,
            generation=stored["index_generation"],
        )
        assert result.outcome == "success"
        assert any("Photosynthesis" in source["content"] for source in result.sources)

        stack["vectors"].delete_by_filename(FILENAME)
        assert stack["vectors"].retrieve("photosynthesis?", FILENAME).outcome == "empty"

    def test_empty_queue_drains_zero(self, stack):
        assert stack["worker"].drain() == 0

    def test_limits_stop_giant_document(self, stack):
        from services.indexing.manifest import manifest_builder
        from services.ingestion.limits import ResourceLimits
        from services.ingestion.worker import IngestionWorker
        from services.parsing.document_parser import DocumentIngestor

        repositories = stack["repositories"]
        repositories.files.create_file_with_job(FILENAME, title="Slow Paper")
        stack["storage"].save(FILENAME, make_pdf_bytes(PAGES))
        tiny = IngestionWorker(
            repositories,
            stack["storage"],
            DocumentIngestor("false", 512, 50),
            stack["vectors"]._embeddings,
            stack["vectors"],
            worker_id="slow-test-worker",
            limits=ResourceLimits(max_pages=0),
            manifest_builder=manifest_builder(stack["settings"]),
        )
        assert tiny.drain() == 1
        latest = repositories.ingestion_jobs.get_latest(FILENAME)
        assert latest["state"] == "failed"
        assert latest["error_category"] == "page_limit_exceeded"
