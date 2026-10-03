"""Unit tests for index staleness and document readability.

Covers ``judge_index``, ``record_stale_reason``, the manifest helpers, and
``refuse``. Everything here is pure comparison logic. Slow work like
embedding a real document or querying Qdrant belongs in tests/slow.
"""

import copy
import json

import pytest

from conftest import make_file_record, make_ingestion_job
from services.indexing.manifest import (
    MISSING_MANIFEST_CHANGE,
    IndexManifest,
    change_details,
    change_labels,
    content_hash,
    manifest_builder,
    manifest_changes,
    manifest_from_json,
    runtime_manifest,
)
from services.indexing.readiness import (
    UNREADABLE_DELETING,
    UNREADABLE_INDEXING,
    UNREADABLE_MISSING,
    UNREADABLE_PENDING,
    UNREADABLE_STALE,
    refuse,
)
from services.indexing.state import (
    PENDING,
    READY,
    STALE,
    judge_index,
    record_stale_reason,
)

pytestmark = pytest.mark.unit


def _processed_record(settings, **overrides):
    manifest = runtime_manifest(
        settings, content_hash_value="abc", index_generation=3
    ).to_json()
    record = make_file_record(
        is_processed=True,
        index_generation=3,
        index_manifest=manifest,
        **overrides,
    )
    return record


class TestJudgeIndex:
    def test_unprocessed_is_pending(self, settings):
        state = judge_index(make_file_record(is_processed=False), settings)
        assert state.state == PENDING
        assert state.is_stale is False

    def test_unprocessed_keeps_generation(self, settings):
        state = judge_index(
            make_file_record(is_processed=False, index_generation=7), settings
        )
        assert state.index_generation == 7

    def test_matching_manifest_is_ready(self, settings):
        state = judge_index(_processed_record(settings), settings)
        assert state.state == READY
        assert state.changes == []
        assert state.is_stale is False

    def test_processed_without_manifest_is_stale(self, settings):
        record = make_file_record(is_processed=True, index_manifest=None)
        state = judge_index(record, settings)
        assert state.state == STALE
        assert state.changes == [MISSING_MANIFEST_CHANGE]

    def test_chunk_size_change_is_stale(self, settings):
        record = _processed_record(settings)
        changed = copy.deepcopy(settings)
        changed.chunking.chunk_size_tokens = 9999
        state = judge_index(record, changed)
        assert state.state == STALE
        assert "chunk_size_tokens" in state.changes

    def test_embedding_model_change_is_stale(self, settings):
        record = _processed_record(settings)
        changed = copy.deepcopy(settings)
        changed.embedding.embedding_model = "other/model"
        assert judge_index(record, changed).is_stale is True

    def test_collection_change_is_stale(self, settings):
        record = _processed_record(settings)
        changed = copy.deepcopy(settings)
        changed.vector.index_name = "other-collection"
        assert judge_index(record, changed).is_stale is True

    def test_reranker_toggle_alone_not_compared_as_runtime(self, settings):
        record = _processed_record(settings)
        changed = copy.deepcopy(settings)
        changed.rerank.enabled = not settings.rerank.enabled
        state = judge_index(record, changed)
        assert state.state == READY

    def test_generation_defaults_to_zero(self, settings):
        record = make_file_record(is_processed=False)
        del record["index_generation"]
        record["index_generation"] = None
        assert judge_index(record, settings).index_generation == 0

    def test_to_dict_shape(self, settings):
        state = judge_index(_processed_record(settings), settings)
        payload = state.to_dict(settings)
        assert payload["state"] == READY
        assert payload["manifest"]["content_hash"] == "abc"
        assert (
            payload["runtime_manifest"]["collection_name"] == settings.vector.index_name
        )
        assert payload["changes"] == []
        assert payload["change_details"] == []


class TestRecordStaleReason:
    class _Files:
        def __init__(self, record):
            self.record = record
            self.calls = []

        def set_index_stale(self, filename, reason):
            self.calls.append((filename, reason))
            self.record["index_stale_reason"] = reason

    def test_sets_reason_when_stale(self, settings):
        record = make_file_record(
            is_processed=True, index_manifest=None, index_stale_reason=None
        )
        files = self._Files(record)
        state = record_stale_reason(files, record["filename"], record, settings)
        assert state.is_stale is True
        assert files.calls != []
        assert record["index_stale_reason"] is not None

    def test_no_write_when_reason_unchanged(self, settings):
        record = _processed_record(settings)
        record["index_stale_reason"] = None
        files = self._Files(record)
        record_stale_reason(files, record["filename"], record, settings)
        assert files.calls == []

    def test_clears_reason_when_back_to_ready(self, settings):
        record = _processed_record(settings)
        record["index_stale_reason"] = "chunk size"
        files = self._Files(record)
        record_stale_reason(files, record["filename"], record, settings)
        assert files.calls == [(record["filename"], None)]


class TestManifestHelpers:
    def test_content_hash_is_sha256(self):
        import hashlib

        assert content_hash(b"hello") == hashlib.sha256(b"hello").hexdigest()

    def test_manifest_roundtrip(self, settings):
        built = runtime_manifest(settings, content_hash_value="h", index_generation=2)
        restored = manifest_from_json(built.to_json())
        assert restored == built

    def test_from_json_rejects_garbage(self):
        assert manifest_from_json(None) is None
        assert manifest_from_json("") is None
        assert manifest_from_json("not json") is None
        assert manifest_from_json("[1,2]") is None
        assert manifest_from_json(json.dumps({"a": 1})) is None

    def test_from_json_rejects_missing_field(self, settings):
        built = runtime_manifest(settings).to_dict()
        built.pop("parser")
        assert manifest_from_json(json.dumps(built)) is None

    def test_changes_empty_when_equal(self, settings):
        manifest = runtime_manifest(settings)
        assert manifest_changes(manifest, runtime_manifest(settings)) == []

    def test_changes_none_stored_is_empty(self, settings):
        assert manifest_changes(None, runtime_manifest(settings)) == []

    def test_changes_lists_each_runtime_field(self, settings):
        stored = runtime_manifest(settings)
        changed_dict = stored.to_dict()
        changed_dict["chunk_size_tokens"] = -1
        changed_dict["embedding_model"] = "other"
        changed = IndexManifest(**changed_dict)
        changes = manifest_changes(changed, runtime_manifest(settings))
        assert "chunk_size_tokens" in changes
        assert "embedding_model" in changes

    def test_reranker_enabled_not_a_runtime_change(self, settings):
        stored = runtime_manifest(settings)
        changed_dict = stored.to_dict()
        changed_dict["reranker_enabled"] = not stored.reranker_enabled
        changed = IndexManifest(**changed_dict)
        assert manifest_changes(changed, runtime_manifest(settings)) == []

    def test_change_labels_user_facing(self):
        assert change_labels(["chunk_size_tokens"]) == ["chunk size"]
        assert change_labels(["unknown-field"]) == ["unknown-field"]

    def test_change_details_pairs_indexed_and_current(self, settings):
        stored = runtime_manifest(settings)
        details = change_details(stored, ["chunk_size_tokens"], settings)
        assert details[0]["field"] == "chunk_size_tokens"
        assert details[0]["indexed"] == stored.chunk_size_tokens
        assert details[0]["current"] == settings.chunking.chunk_size_tokens

    def test_change_details_without_stored_is_none(self, settings):
        details = change_details(None, ["chunk_size_tokens"], settings)
        assert details[0]["indexed"] is None

    def test_builder_records_bytes_and_generation(self, settings):
        build = manifest_builder(settings)
        manifest = build(b"bytes", 5)
        assert manifest.content_hash == content_hash(b"bytes")
        assert manifest.index_generation == 5


class TestRefuse:
    def test_missing_record(self, settings):
        refusal = refuse(None, None, settings)
        assert refusal is not None
        assert refusal.category == UNREADABLE_MISSING
        assert refusal.status == 404

    def test_deleting_record_blocked(self, settings):
        for state in ("deleting", "delete_failed"):
            record = make_file_record(deletion_state=state)
            refusal = refuse(record, None, settings)
            assert refusal is not None
            assert refusal.category == UNREADABLE_DELETING
            assert refusal.status == 409

    @pytest.mark.parametrize("job_state", ["queued", "running", "cancelling"])
    def test_reindex_in_flight_blocked(self, settings, job_state):
        record = _processed_record(settings)
        refusal = refuse(record, make_ingestion_job(job_state), settings)
        assert refusal is not None
        assert refusal.category == UNREADABLE_INDEXING
        assert refusal.detail["job"]["state"] == job_state

    @pytest.mark.parametrize("job_state", ["succeeded", "failed", "cancelled"])
    def test_terminal_job_does_not_block(self, settings, job_state):
        record = _processed_record(settings)
        assert refuse(record, make_ingestion_job(job_state), settings) is None

    def test_unprocessed_without_job_is_pending(self, settings):
        record = make_file_record(is_processed=False)
        refusal = refuse(record, None, settings)
        assert refusal is not None
        assert refusal.category == UNREADABLE_PENDING

    def test_stale_manifest_refused_with_reindex_action(self, settings):
        record = make_file_record(is_processed=True, index_manifest=None)
        refusal = refuse(record, None, settings)
        assert refusal is not None
        assert refusal.category == UNREADABLE_STALE
        assert refusal.detail["action"] == "reindex"
        assert refusal.detail["index"]["state"] == STALE

    def test_ready_record_readable(self, settings):
        assert refuse(_processed_record(settings), None, settings) is None

    def test_unprocessed_with_active_job_still_pending(self, settings):
        record = make_file_record(is_processed=False)
        refusal = refuse(record, make_ingestion_job("running"), settings)
        assert refusal is not None
        assert refusal.category == UNREADABLE_PENDING
