"""Build an ingestion worker over an app's injected services."""

from typing import Any

from services.ingestion.worker import IngestionWorker


def build_test_worker(app: Any, worker_id: str = "test-worker") -> IngestionWorker:
    """Return a worker wired to the dependencies injected into the app."""
    return IngestionWorker(
        repositories=app.config["TEST_REPOSITORIES"],
        storage=app.config["TEST_STORAGE"],
        parser=app.config["TEST_PARSER"],
        embedding_service=app.config["TEST_EMBEDDINGS"],
        vector_service=app.config["TEST_VECTORS"],
        worker_id=worker_id,
    )
