"""Unit tests for brief scope resolution and cancellation.

Scope checks run against stub repositories. Cancellation uses the real
process-local registry with a reset fixture. Run with plain ``pytest``.
"""

import pytest

from conftest import make_file_record, make_ingestion_job
from services.brief import cancellation
from services.brief.scope import (
    BRIEF_DOCUMENT_COUNT,
    BriefScope,
    ScopedDocument,
    resolve_scope,
)
from services.indexing.manifest import runtime_manifest

pytestmark = pytest.mark.unit


@pytest.fixture()
def _clean_registry():
    cancellation.reset()
    yield
    cancellation.reset()


def _ready_record(settings, filename="a.pdf", doc_id="file-a", title="Paper A"):
    manifest = runtime_manifest(
        settings, content_hash_value="h", index_generation=1
    ).to_json()
    return make_file_record(
        id=doc_id,
        filename=filename,
        title=title,
        is_processed=True,
        index_generation=1,
        index_manifest=manifest,
    )


class StubFiles:
    def __init__(self, records):
        self._records = dict(records)
        self.stale_calls = []

    def get_file(self, filename):
        return self._records.get(filename)

    def set_index_stale(self, filename, reason):
        self.stale_calls.append((filename, reason))


class StubRepos:
    def __init__(self, records):
        self.files = StubFiles(records)
        self._jobs = {}

    def jobs_for(self, filename, job):
        self._jobs[filename] = job

    @property
    def ingestion_jobs(self):
        outer = self

        class Jobs:
            def get_latest(self, filename):
                return outer._jobs.get(filename)

        return Jobs()


class TestBriefScopeValue:
    def _scope(self):
        return BriefScope(
            documents=(
                ScopedDocument("A", "a.pdf", "id-a", "Paper A", 1),
                ScopedDocument("B", "b.pdf", "id-b", "Paper B", 2),
            ),
            filenames=("a.pdf", "b.pdf"),
        )

    def test_labels_and_lookup(self):
        scope = self._scope()
        assert scope.labels == ("A", "B")
        assert scope.document_for_label("a").filename == "a.pdf"
        assert scope.document_for_label(" B ") is not None
        assert scope.document_for_label("C") is None

    def test_describe_hides_filenames(self):
        described = self._scope().describe()
        assert "a.pdf" not in described
        assert "[A] Paper A" in described

    def test_to_dict(self):
        (first, _) = self._scope().to_dict()
        assert first["label"] == "A"
        assert "filename" not in first


class TestResolveScope:
    def test_happy_path(self, settings):
        repos = StubRepos(
            {
                "a.pdf": _ready_record(settings),
                "b.pdf": _ready_record(settings, "b.pdf", "file-b", "Paper B"),
            }
        )
        scope, refusal = resolve_scope(
            repositories=repos, settings=settings, filenames=["a.pdf", "b.pdf"]
        )
        assert refusal is None
        assert scope is not None
        assert scope.filenames == ("a.pdf", "b.pdf")
        assert scope.documents[0].title == "Paper A"

    def test_wrong_count_refused(self, settings):
        repos = StubRepos({})
        _, refusal = resolve_scope(
            repositories=repos, settings=settings, filenames=["a.pdf"]
        )
        assert refusal is not None
        assert refusal["category"] == "brief_scope_invalid"
        assert refusal["status"] == 400

    def test_duplicates_refused(self, settings):
        repos = StubRepos({})
        _, refusal = resolve_scope(
            repositories=repos, settings=settings, filenames=["a.pdf", "a.pdf"]
        )
        assert refusal["category"] == "brief_scope_invalid"

    def test_too_many_refused(self, settings):
        repos = StubRepos({})
        _, refusal = resolve_scope(
            repositories=repos, settings=settings, filenames=["a.pdf", "b.pdf", "c.pdf"]
        )
        assert refusal is not None

    def test_blank_names_ignored(self, settings):
        repos = StubRepos(
            {
                "a.pdf": _ready_record(settings),
                "b.pdf": _ready_record(settings, "b.pdf", "file-b", "Paper B"),
            }
        )
        scope, refusal = resolve_scope(
            repositories=repos, settings=settings, filenames=["a.pdf", "  ", "b.pdf"]
        )
        assert refusal is None
        assert scope is not None

    def test_unknown_document_names_label(self, settings):
        repos = StubRepos({"a.pdf": _ready_record(settings)})
        _, refusal = resolve_scope(
            repositories=repos, settings=settings, filenames=["a.pdf", "missing.pdf"]
        )
        assert refusal is not None
        assert refusal["label"] == "B"
        assert refusal["status"] == 404

    def test_stale_document_refused(self, settings):
        stale = make_file_record(
            id="file-b",
            filename="b.pdf",
            title="B",
            is_processed=True,
            index_manifest=None,
        )
        repos = StubRepos({"a.pdf": _ready_record(settings), "b.pdf": stale})
        _, refusal = resolve_scope(
            repositories=repos, settings=settings, filenames=["a.pdf", "b.pdf"]
        )
        assert refusal["category"] == "index_stale"

    def test_reindexing_document_refused(self, settings):
        repos = StubRepos(
            {
                "a.pdf": _ready_record(settings),
                "b.pdf": _ready_record(settings, "b.pdf", "file-b", "Paper B"),
            }
        )
        repos.jobs_for("b.pdf", make_ingestion_job("running"))
        _, refusal = resolve_scope(
            repositories=repos, settings=settings, filenames=["a.pdf", "b.pdf"]
        )
        assert refusal["category"] == "document_indexing"

    def test_deleting_document_refused(self, settings):
        deleting = _ready_record(settings, "b.pdf", "file-b", "Paper B")
        deleting["deletion_state"] = "deleting"
        repos = StubRepos({"a.pdf": _ready_record(settings), "b.pdf": deleting})
        _, refusal = resolve_scope(
            repositories=repos, settings=settings, filenames=["a.pdf", "b.pdf"]
        )
        assert refusal["category"] == "document_deleting"

    def test_document_count_constant(self):
        assert BRIEF_DOCUMENT_COUNT == 2


class TestCancellation:
    def test_cancel_unknown_is_false(self, _clean_registry):
        assert cancellation.cancel("nope") is False

    def test_register_cancel_release(self, _clean_registry):
        event = cancellation.register("brief-1")
        assert cancellation.cancel("brief-1") is True
        assert event.is_set() is True
        cancellation.release("brief-1")
        assert cancellation.cancel("brief-1") is False

    def test_release_missing_is_safe(self, _clean_registry):
        cancellation.release("never-registered")

    def test_reset_forgets_all(self, _clean_registry):
        cancellation.register("brief-1")
        cancellation.reset()
        assert cancellation.cancel("brief-1") is False

    def test_reregister_replaces_slot(self, _clean_registry):
        first = cancellation.register("brief-1")
        second = cancellation.register("brief-1")
        assert second is not first
        assert cancellation.cancel("brief-1") is True
