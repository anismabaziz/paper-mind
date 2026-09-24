"""Vector store hardening tests."""

import threading
from unittest.mock import MagicMock

import pytest

import providers
import settings as settings_module
from services.retrieval.base import VectorDimensionError
from services.retrieval.qdrant_store import QdrantIndexAdapter, _MAX_CACHE_ENTRIES


def test_dimension_mismatch_raises_typed_error_and_preserves_prior(tmp_path):
    """Mismatch leaves prior points intact, no collection wipe or cache clear."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_client.upsert.return_value = None

    adapter = QdrantIndexAdapter(mock_client, "pdf-index")
    ok_vectors = [
        {
            "id": "id-ok",
            "values": [0.1] * 1024,
            "metadata": {"pdf_name": "docA.pdf", "content": "hello"},
            "sparse_vector": {"indices": [1], "values": [0.5]},
        }
    ]
    adapter.upsert(ok_vectors)
    prior_payload = dict(adapter._payload_by_id)
    prior_sparse = dict(adapter._sparse_by_id)
    assert len(prior_payload) == 1

    def raise_dim(*_a, **_k):
        raise Exception("Vector size mismatch: expected 1024 got 512")

    mock_client.upsert.side_effect = raise_dim
    bad_vectors = [
        {
            "id": "id-bad",
            "values": [0.1] * 512,
            "metadata": {"pdf_name": "docB.pdf", "content": "bad"},
            "sparse_vector": {"indices": [2], "values": [0.6]},
        }
    ]
    with pytest.raises(VectorDimensionError) as exc:
        adapter.upsert(bad_vectors)
    assert "POST /delete-embeddings" in str(exc.value)
    assert mock_client.delete_collection.call_count == 0
    assert adapter._payload_by_id == prior_payload
    assert adapter._sparse_by_id == prior_sparse


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


def test_caches_bounded_and_only_filtered_deletes_prune():
    """Caches capped at MAX and only filtered deletes mutate them."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_client.upsert.return_value = None

    adapter = QdrantIndexAdapter(mock_client, "pdf-index")
    for i in range(_MAX_CACHE_ENTRIES + 50):
        vid = f"vid-{i}"
        adapter._payload_by_id[vid] = {"pdf_name": f"doc{i % 10}.pdf"}
        adapter._sparse_by_id[vid] = {"indices": [i], "values": [1.0]}
    adapter._enforce_cache_bounds()
    assert len(adapter._payload_by_id) <= _MAX_CACHE_ENTRIES
    assert len(adapter._sparse_by_id) <= _MAX_CACHE_ENTRIES

    adapter._payload_by_id = {
        "a": {"pdf_name": "doc1.pdf"},
        "b": {"pdf_name": "doc2.pdf"},
        "c": {"pdf_name": "doc1.pdf"},
    }
    adapter._sparse_by_id = {"a": {}, "b": {}, "c": {}}
    mock_client.delete.return_value = None
    adapter.delete(filter={"pdf_name": "doc1.pdf"})
    assert "b" in adapter._payload_by_id and "a" not in adapter._payload_by_id
    assert "b" in adapter._sparse_by_id and "a" not in adapter._sparse_by_id


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
