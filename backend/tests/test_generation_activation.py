"""
Atomic index generation activation tests.

A generation becomes the Document's active index only after every stored
Passage passes validation: the expected chunk count, dense vectors, sparse
vectors, payload metadata, and Page provenance. The vector store reports what
a generation actually contains; the validation policy lives with the
retrieval service.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from services.retrieval.base import VectorStoreConfigurationError
from services.retrieval.hybrid import SPARSE_METHOD, TOKENIZER_VERSION
from services.retrieval.qdrant_store import QdrantIndexAdapter
from services.retrieval.vector_service import VectorService


def _valid_collection():
    return SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors={"dense": SimpleNamespace(size=1024, distance="Cosine")},
                sparse_vectors={"sparse": SimpleNamespace(modifier="idf")},
            )
        )
    )


def _ready_client(points, total=None):
    """Build a mocked Qdrant client holding one generation's points."""
    client = MagicMock()
    client.collection_exists.return_value = True
    client.get_collection.return_value = _valid_collection()
    client.count.return_value = SimpleNamespace(
        count=len(points) if total is None else total
    )
    client.scroll.return_value = (points, None)
    return client


def _point(index, page_no=1, content="passage text", **payload):
    metadata = {
        "content": content,
        "pdf_name": "doc.pdf",
        "chunk_index": index,
        "content_hash": f"hash-{index}",
        "page_no": page_no,
        "sparse_method": SPARSE_METHOD,
        "sparse_tokenizer_version": TOKENIZER_VERSION,
        "index_generation": 2,
    }
    metadata.update(payload)
    return SimpleNamespace(
        id=f"point-{index}",
        payload=metadata,
        vector={
            "dense": [0.1] * 1024,
            "sparse": SimpleNamespace(indices=[7], values=[1.0]),
        },
    )


def test_generation_report_counts_vectors_payload_and_pages():
    """The report aggregates what one generation actually stores."""
    client = _ready_client([_point(0, page_no=2), _point(1, page_no=3)])
    adapter = QdrantIndexAdapter(client, "pdf-index")

    report = adapter.generation_report(
        {"pdf_name": "doc.pdf", "index_generation": 2},
        limit=10,
        value_keys=("sparse_method", "sparse_tokenizer_version"),
    )

    assert report["total"] == 2
    assert report["without_dense"] == 0
    assert report["without_sparse"] == 0
    assert report["missing_payload"] == {}
    assert report["empty_payload"] == {}
    assert report["pages"] == [2, 3]
    assert report["truncated"] is False
    assert report["distinct_values"]["sparse_method"] == [SPARSE_METHOD]
    assert report["distinct_values"]["sparse_tokenizer_version"] == [TOKENIZER_VERSION]
    assert client.scroll.call_args.kwargs["scroll_filter"].must


def test_generation_report_flags_missing_representations():
    """Points without a dense or sparse vector are reported separately."""
    dense_only = _point(0)
    dense_only.vector = {"dense": [0.1] * 1024}
    sparse_only = _point(1)
    sparse_only.vector = {
        "sparse": SimpleNamespace(indices=[7], values=[1.0]),
    }
    adapter = QdrantIndexAdapter(_ready_client([dense_only, sparse_only]), "pdf-index")

    report = adapter.generation_report({"pdf_name": "doc.pdf"}, limit=10)

    assert report["without_sparse"] == 1
    assert report["without_dense"] == 1


def test_generation_report_flags_an_empty_sparse_vector():
    """A stored but empty sparse representation is not usable."""
    empty = _point(0)
    empty.vector = {
        "dense": [0.1] * 1024,
        "sparse": SimpleNamespace(indices=[], values=[]),
    }
    adapter = QdrantIndexAdapter(_ready_client([empty]), "pdf-index")

    report = adapter.generation_report({"pdf_name": "doc.pdf"}, limit=10)

    assert report["without_sparse"] == 1


def test_generation_report_flags_absent_and_empty_payload_values():
    """Absent and blank payload values are counted per field."""
    missing = _point(0)
    missing.payload.pop("page_no")
    blank = _point(1, content="   ")
    adapter = QdrantIndexAdapter(_ready_client([missing, blank]), "pdf-index")

    report = adapter.generation_report({"pdf_name": "doc.pdf"}, limit=10)

    assert report["missing_payload"] == {"page_no": 1}
    assert report["empty_payload"] == {"content": 1}


def test_generation_report_marks_a_scrolled_subset_as_truncated():
    """More stored points than the scan window is reported as truncated."""
    client = _ready_client([_point(0)], total=40)
    adapter = QdrantIndexAdapter(client, "pdf-index")

    report = adapter.generation_report({"pdf_name": "doc.pdf"}, limit=10)

    assert report["total"] == 40
    assert report["truncated"] is True


class ReportingStore:
    """Vector store stand-in returning a fixed generation report."""

    def __init__(self, **report):
        """Initialize."""
        self.reports = []
        self.report = {
            "total": 1,
            "without_dense": 0,
            "without_sparse": 0,
            "missing_payload": {},
            "empty_payload": {},
            "pages": [1],
            "truncated": False,
            "distinct_values": {
                "sparse_method": [SPARSE_METHOD],
                "sparse_tokenizer_version": [TOKENIZER_VERSION],
            },
        }
        self.report.update(report)

    def generation_report(self, filter, limit, value_keys=()):
        """Return the prepared report and remember the filter it was asked for."""
        self.reports.append((filter, limit))
        return self.report

    def query(self, **kwargs):
        """Return no matches."""
        return {"matches": [], "method": "dense", "outcome": "empty"}

    def delete(self, filter=None, delete_all=False):
        """Do delete."""
        return {"deleted": 0}


def _service(**report):
    store = ReportingStore(**report)
    return VectorService(store), store


def test_validation_passes_a_complete_generation():
    """A complete generation validates and reports what it checked."""
    service, store = _service()

    report = service.validate_generation("doc.pdf", 2, expected_count=1, page_count=4)

    assert report["total"] == 1
    assert store.reports[0][0] == {"pdf_name": "doc.pdf", "index_generation": 2}


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        ({"total": 0}, "expected 1"),
        ({"total": 2}, "expected 1"),
        ({"truncated": True}, "not fully inspected"),
        ({"without_dense": 1}, "without a dense vector"),
        ({"without_sparse": 1}, "without a sparse vector"),
        ({"missing_payload": {"page_no": 1}}, "missing payload for page_no"),
        ({"empty_payload": {"content": 1}}, "blank content"),
        ({"pages": []}, "no Page provenance"),
        ({"pages": [0, 9]}, "outside the document's 4 Pages"),
        (
            {"distinct_values": {"sparse_method": ["other-v1"]}},
            "indexed with sparse method",
        ),
        ({"distinct_values": {}}, "no sparse method was recorded"),
    ],
)
def test_validation_rejects_an_incomplete_generation(report, expected):
    """Every stage of a generation is checked before it can be activated."""
    service, _ = _service(**report)

    with pytest.raises(VectorStoreConfigurationError) as raised:
        service.validate_generation("doc.pdf", 2, expected_count=1, page_count=4)

    assert "Index generation 2" in str(raised.value)
    assert expected in str(raised.value)


def test_validation_rejects_an_unexpected_sparse_tokenizer():
    """A generation indexed by another tokenizer version is not activated."""
    service, _ = _service(
        distinct_values={
            "sparse_method": [SPARSE_METHOD],
            "sparse_tokenizer_version": ["other-tokenizer"],
        }
    )

    with pytest.raises(VectorStoreConfigurationError) as raised:
        service.validate_generation("doc.pdf", 2, expected_count=1)

    assert "tokenizer" in str(raised.value)


def test_validation_skips_page_bounds_when_no_page_count_is_known():
    """Documents without a known Page count skip only the page range check."""
    service, _ = _service(pages=[7])

    report = service.validate_generation("doc.pdf", 2, expected_count=1)

    assert report["pages"] == [7]
