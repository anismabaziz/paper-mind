"""Unit tests for generation activation validation.

The vector store is faked. Run with plain ``pytest``.
"""

import pytest

from services.indexing.activation import activate_generation
from services.retrieval.base import VectorStoreConfigurationError
from services.retrieval.hybrid import SPARSE_METHOD, TOKENIZER_VERSION

pytestmark = pytest.mark.unit


def _report(**overrides):
    report = {
        "total": 3,
        "truncated": False,
        "inspected": 3,
        "without_dense": 0,
        "without_sparse": 0,
        "missing_payload": {},
        "empty_payload": {},
        "pages": [1, 2, 3],
        "distinct_values": {
            "sparse_method": [SPARSE_METHOD],
            "sparse_tokenizer_version": [TOKENIZER_VERSION],
        },
    }
    report.update(overrides)
    return report


class FakeStore:
    def __init__(self, report):
        self._report = report
        self.count_calls = []

    def generation_report(self, filter=None, limit=1000, value_keys=()):
        return self._report() if callable(self._report) else self._report

    def count(self, filter=None):
        self.count_calls.append(filter)
        return 3


class FakeFiles:
    def __init__(self, record):
        self._record = record

    def get_file(self, filename):
        return self._record


class FakeJobs:
    def __init__(self):
        self.ready_calls = []

    def mark_ready(self, job_id, worker_id, index_generation, index_manifest):
        self.ready_calls.append((job_id, worker_id, index_generation))
        return {"id": job_id, "state": "succeeded"}


class FakeCleanups:
    def __init__(self):
        self.scheduled = []

    def schedule(self, file_id, filename, generation):
        self.scheduled.append((file_id, filename, generation))


def _job(generation=2):
    return {"id": "job-1", "filename": "a.pdf", "generation": generation}


def _activate(store, record="default", expected_count=3):
    if record == "default":
        record = {"id": "file-1", "filename": "a.pdf", "index_generation": 1}
    files = FakeFiles(record)
    jobs = FakeJobs()
    cleanups = FakeCleanups()
    out = activate_generation(
        files=files,
        jobs=jobs,
        cleanups=cleanups,
        vector_store=store,
        job=_job(),
        worker_id="w1",
        expected_count=expected_count,
        page_count=3,
        manifest_json='{"v": 1}',
    )
    return out, jobs, cleanups


class TestActivateGeneration:
    def test_happy_path_schedules_cleanup_then_marks_ready(self):
        out, jobs, cleanups = _activate(FakeStore(_report()))
        assert out == {"id": "job-1", "state": "succeeded"}
        assert cleanups.scheduled == [("file-1", "a.pdf", 1)]
        assert jobs.ready_calls == [("job-1", "w1", 2)]

    def test_missing_document_returns_none(self):
        out, jobs, cleanups = _activate(FakeStore(_report()), record=None)
        assert out is None
        assert jobs.ready_calls == []

    def test_count_mismatch_rejected(self):
        with pytest.raises(VectorStoreConfigurationError, match="expected 3 vectors"):
            _activate(FakeStore(_report(total=2)))

    def test_missing_dense_rejected(self):
        with pytest.raises(VectorStoreConfigurationError, match="dense"):
            _activate(FakeStore(_report(without_dense=1)))

    def test_missing_sparse_rejected(self):
        with pytest.raises(VectorStoreConfigurationError, match="sparse"):
            _activate(FakeStore(_report(without_sparse=2)))

    def test_missing_payload_rejected(self):
        with pytest.raises(VectorStoreConfigurationError, match="content_hash"):
            _activate(FakeStore(_report(missing_payload={"content_hash": 2})))

    def test_blank_payload_rejected(self):
        with pytest.raises(VectorStoreConfigurationError, match="blank"):
            _activate(FakeStore(_report(empty_payload={"content": 1})))

    def test_no_page_provenance_rejected(self):
        with pytest.raises(VectorStoreConfigurationError, match="Page provenance"):
            _activate(FakeStore(_report(pages=[])))

    def test_out_of_range_pages_rejected(self):
        with pytest.raises(VectorStoreConfigurationError, match="outside"):
            _activate(FakeStore(_report(pages=[1, 99])))

    def test_wrong_sparse_method_rejected(self):
        report = _report()
        report["distinct_values"] = {
            "sparse_method": ["other"],
            "sparse_tokenizer_version": [TOKENIZER_VERSION],
        }
        with pytest.raises(VectorStoreConfigurationError, match="sparse method"):
            _activate(FakeStore(report))

    def test_missing_sparse_setting_rejected(self):
        report = _report()
        report["distinct_values"] = {}
        with pytest.raises(VectorStoreConfigurationError, match="no sparse method"):
            _activate(FakeStore(report))

    def test_truncated_report_rejected(self):
        with pytest.raises(VectorStoreConfigurationError, match="not fully inspected"):
            _activate(FakeStore(_report(truncated=True, inspected=1)))

    def test_store_without_reports_falls_back_to_count(self):
        class CountOnly:
            def generation_report(self, filter=None, limit=1000, value_keys=()):
                return None

            def count(self, filter=None):
                return 3

        out, _, _ = _activate(CountOnly())
        assert out is not None

    def test_count_fallback_mismatch_rejected(self):
        class NoReport:
            def generation_report(self, filter=None, limit=1000, value_keys=()):
                return None

            def count(self, filter=None):
                return 1

        with pytest.raises(VectorStoreConfigurationError, match="incomplete"):
            _activate(NoReport())

    def test_zero_expected_count_skips_page_check(self):
        out, _, _ = _activate(FakeStore(_report(total=0, pages=[])), expected_count=0)
        assert out is not None
