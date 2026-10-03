"""Unit tests for batch concurrency and local file storage.

Concurrency spawns real threads but finishes in milliseconds. Storage uses a
temp dir. Run with plain ``pytest``.
"""

import pytest

from services.concurrency import map_batches_concurrently
from storage import LocalStorage

pytestmark = pytest.mark.unit


class TestMapBatches:
    def test_empty(self):
        assert map_batches_concurrently([], lambda b: b, label="t") == []

    def test_single_batch_no_pool(self):
        calls = []
        out = map_batches_concurrently(
            ["only"], lambda b: calls.append(b) or "done", label="t"
        )
        assert out == ["done"]
        assert calls == ["only"]

    def test_order_preserved(self):
        import time

        def slow(batch):
            time.sleep(0.01 * (5 - batch))
            return batch * 10

        assert map_batches_concurrently([1, 2, 3, 4], slow, label="t") == [
            10,
            20,
            30,
            40,
        ]

    def test_before_batch_called(self):
        seen = []
        map_batches_concurrently(
            ["a"], lambda b: b, label="t", before_batch=lambda: seen.append(1)
        )
        assert seen == [1]

    def test_business_error_propagates(self):
        def boom(batch):
            raise ValueError("bad batch")

        with pytest.raises(ValueError, match="bad batch"):
            map_batches_concurrently(["a", "b"], boom, label="t")


class TestLocalStorage:
    @pytest.fixture()
    def storage(self, tmp_path):
        return LocalStorage(tmp_path / "store")

    def test_roundtrip(self, storage):
        storage.save("doc.pdf", b"%PDF-bytes")
        assert storage.open("doc.pdf") == b"%PDF-bytes"
        assert storage.exists("doc.pdf") is True

    def test_missing_is_not_exists(self, storage):
        assert storage.exists("missing.pdf") is False

    def test_delete_missing_ok(self, storage):
        storage.delete("missing.pdf")

    def test_delete_removes(self, storage):
        storage.save("doc.pdf", b"x")
        storage.delete("doc.pdf")
        assert storage.exists("doc.pdf") is False

    def test_list_sorted_with_sizes(self, storage):
        storage.save("b.pdf", b"12345")
        storage.save("a.pdf", b"12")
        assert storage.list() == [
            {"name": "a.pdf", "size": 2},
            {"name": "b.pdf", "size": 5},
        ]

    def test_url_shape(self, storage):
        assert storage.url("doc.pdf") == "/storage/doc.pdf"

    @pytest.mark.parametrize(
        "name", ["", "   ", "../x.pdf", "/abs.pdf", "a/../../x.pdf"]
    )
    def test_traversal_rejected(self, storage, name):
        with pytest.raises(ValueError):
            storage.save(name, b"x")

    def test_overwrite_replaces_bytes(self, storage):
        storage.save("doc.pdf", b"one")
        storage.save("doc.pdf", b"two!")
        assert storage.open("doc.pdf") == b"two!"
