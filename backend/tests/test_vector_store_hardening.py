"""Vector store hardening tests."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import providers
import settings as settings_module
from services.retrieval.base import (
    VectorDimensionError,
    VectorStoreConfigurationError,
    VectorStoreUnavailableError,
)
from services.retrieval.qdrant_store import QdrantIndexAdapter


def _valid_collection():
    return SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors={"dense": SimpleNamespace(size=1024, distance="Cosine")},
                sparse_vectors={"sparse": SimpleNamespace(modifier="idf")},
            )
        )
    )


def test_upsert_stores_named_dense_and_sparse_vectors():
    """Each stored Qdrant point contains both named representations."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_client.get_collection.return_value = _valid_collection()
    adapter = QdrantIndexAdapter(mock_client, "pdf-index")

    adapter.upsert(
        [
            {
                "id": "id-ok",
                "values": [0.1] * 1024,
                "metadata": {"pdf_name": "doc.pdf", "content": "hello"},
                "sparse_vector": {"indices": [7], "values": [1.0]},
            }
        ]
    )

    point = mock_client.upsert.call_args.kwargs["points"][0]
    assert set(point.vector) == {"dense", "sparse"}
    assert point.vector["dense"] == [0.1] * 1024
    assert point.vector["sparse"].indices == [7]
    assert point.vector["sparse"].values == [1.0]


def test_collection_schema_names_dense_and_sparse_vectors():
    """Collection creation declares explicit dense and sparse names."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = False
    adapter = QdrantIndexAdapter(mock_client, "pdf-index")

    adapter.upsert(
        [
            {
                "id": "id-ok",
                "values": [0.1] * 1024,
                "metadata": {"pdf_name": "doc.pdf", "content": "hello"},
                "sparse_vector": {"indices": [7], "values": [1.0]},
            }
        ]
    )

    vectors_config = mock_client.create_collection.call_args.kwargs["vectors_config"]
    sparse_config = mock_client.create_collection.call_args.kwargs[
        "sparse_vectors_config"
    ]
    assert set(vectors_config) == {"dense"}
    assert set(sparse_config) == {"sparse"}


def test_hybrid_query_uses_named_vectors_rrf_60_and_candidate_depth():
    """Hybrid retrieval sends the shared RRF setting and candidate depth."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_client.get_collection.return_value = _valid_collection()
    mock_client.query_points.return_value.points = [
        {
            "id": "point-1",
            "score": 0.5,
            "payload": {"pdf_name": "doc.pdf", "content": "hello"},
        }
    ]
    adapter = QdrantIndexAdapter(mock_client, "pdf-index")

    result = adapter.query(
        [0.1, 0.2],
        50,
        filter={"pdf_name": "doc.pdf"},
        sparse_vector={"indices": [7], "values": [1.0]},
    )

    assert result["matches"][0]["id"] == "point-1"
    assert result["method"] == "hybrid"
    assert result["outcome"] == "success"
    call = mock_client.query_points.call_args.kwargs
    assert [prefetch.using for prefetch in call["prefetch"]] == ["dense", "sparse"]
    assert [prefetch.limit for prefetch in call["prefetch"]] == [50, 50]
    assert call["query"].rrf.k == 60
    assert call["limit"] == 50


def test_sparse_query_selects_named_sparse_vector():
    """Sparse-only retrieval selects the stored sparse representation."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_client.get_collection.return_value = _valid_collection()
    mock_client.query_points.return_value.points = [
        {
            "id": "point-1",
            "score": 2.0,
            "payload": {"pdf_name": "doc.pdf", "content": "hello"},
        }
    ]
    adapter = QdrantIndexAdapter(mock_client, "pdf-index")

    result = adapter.query(
        [0.1, 0.2],
        50,
        filter={"pdf_name": "doc.pdf"},
        method="sparse",
        sparse_vector={"indices": [7], "values": [1.0]},
    )

    assert result["matches"][0]["id"] == "point-1"
    assert result["method"] == "sparse"
    assert result["outcome"] == "success"
    call = mock_client.query_points.call_args.kwargs
    assert call["using"] == "sparse"
    assert call["query"].indices == [7]
    assert "prefetch" not in call


def test_empty_qdrant_result_is_recorded_as_empty_evidence():
    """A successful Qdrant query with no points is not an outage."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_client.get_collection.return_value = _valid_collection()
    mock_client.query_points.return_value.points = []
    adapter = QdrantIndexAdapter(mock_client, "pdf-index")

    result = adapter.query([0.1, 0.2], 50, method="dense")

    assert result == {"matches": [], "method": "dense", "outcome": "empty"}


def test_query_rejects_a_missing_collection_instead_of_creating_empty_evidence():
    """A query cannot turn a missing collection into genuine empty evidence."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = False
    adapter = QdrantIndexAdapter(mock_client, "pdf-index")

    with pytest.raises(VectorStoreConfigurationError):
        adapter.query([0.1] * 1024, 50, method="dense")

    mock_client.create_collection.assert_not_called()
    mock_client.query_points.assert_not_called()


def test_qdrant_authentication_failure_is_a_configuration_error():
    """Rejected Qdrant credentials remain distinct from an outage."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_client.get_collection.return_value = _valid_collection()
    mock_client.query_points.side_effect = RuntimeError(
        "Unexpected Response: 401 Unauthorized"
    )
    adapter = QdrantIndexAdapter(mock_client, "pdf-index")

    with pytest.raises(VectorStoreConfigurationError):
        adapter.query([0.1] * 1024, 50, method="dense")


def test_qdrant_query_outage_is_not_reported_as_empty_evidence():
    """A transport failure raises the unavailable outcome."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_client.get_collection.return_value = _valid_collection()
    mock_client.query_points.side_effect = RuntimeError("connection refused")
    adapter = QdrantIndexAdapter(mock_client, "pdf-index")

    with pytest.raises(VectorStoreUnavailableError):
        adapter.query([0.1, 0.2], 50, method="dense")


def test_invalid_existing_collection_schema_is_a_configuration_error():
    """An incompatible persisted collection is rejected before querying."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_client.get_collection.return_value = _valid_collection()
    mock_client.get_collection.return_value.config.params.vectors = {}
    mock_client.get_collection.return_value.config.params.sparse_vectors = {}
    adapter = QdrantIndexAdapter(mock_client, "pdf-index")

    with pytest.raises(VectorStoreConfigurationError):
        adapter.query([0.1, 0.2], 50, method="dense")


def test_query_dimension_mismatch_keeps_its_typed_error():
    """A query dimension failure remains distinct from service unavailability."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_client.get_collection.return_value = _valid_collection()
    mock_client.query_points.side_effect = RuntimeError(
        "Wrong input: Vector dimension error: expected 1024 got 2"
    )
    adapter = QdrantIndexAdapter(mock_client, "pdf-index")

    with pytest.raises(VectorDimensionError) as exc:
        adapter.query([0.1, 0.2], 50, method="dense")

    assert exc.value.expected == 1024
    assert exc.value.got == 2


def test_dimension_mismatch_raises_typed_error_without_wiping_collection():
    """A rejected upsert never deletes the existing collection."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_client.get_collection.return_value = _valid_collection()
    mock_client.upsert.return_value = None

    adapter = QdrantIndexAdapter(mock_client, "pdf-index")
    adapter.upsert(
        [
            {
                "id": "id-ok",
                "values": [0.1] * 1024,
                "metadata": {"pdf_name": "docA.pdf", "content": "hello"},
                "sparse_vector": {"indices": [1], "values": [0.5]},
            }
        ]
    )

    def raise_dim(*_a, **_k):
        raise Exception("Vector size mismatch: expected 1024 got 512")

    mock_client.upsert.side_effect = raise_dim
    with pytest.raises(VectorDimensionError) as exc:
        adapter.upsert(
            [
                {
                    "id": "id-bad",
                    "values": [0.1] * 1024,
                    "metadata": {"pdf_name": "docB.pdf", "content": "bad"},
                    "sparse_vector": {"indices": [2], "values": [0.6]},
                }
            ]
        )
    assert "POST /delete-embeddings" in str(exc.value)
    assert mock_client.upsert.call_count == 2
    assert mock_client.delete_collection.call_count == 0


def test_concurrent_ensure_does_not_double_create():
    """Concurrent first requests guard collection ensure with a lock."""
    mock_client = MagicMock()
    calls: list[int] = []

    def exists_side(*_a, **_k):
        import time

        time.sleep(0.02)
        return False

    def create_side(*_a, **_k):
        calls.append(1)
        import time

        time.sleep(0.01)
        return None

    mock_client.collection_exists.side_effect = exists_side
    mock_client.create_collection.side_effect = create_side

    adapter = QdrantIndexAdapter(mock_client, "pdf-index")
    threads = [threading.Thread(target=adapter._ensure_collection) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(calls) == 1
    assert adapter._ensured is True


def test_singletons_guarded_against_concurrent_access():
    """Settings singleton built once under concurrent first access."""
    settings_module.set_settings(None)
    providers.reset_providers()
    results: list[object] = []

    def get_s():
        results.append(settings_module.get_settings())

    threads = [threading.Thread(target=get_s) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 20
    assert all(r is results[0] for r in results)

    # cleanup for other tests (conftest autouse will reinstall TEST_SETTINGS)
    settings_module.set_settings(None)
    providers.reset_providers()
