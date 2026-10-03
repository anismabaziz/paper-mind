"""Unit tests for the brief evidence ledger.

Ids must stay stable across searches. Run with plain ``pytest``.
"""

import pytest

from services.brief.evidence import (
    EVIDENCE_PREFIX,
    EXCERPT_CHARS,
    PAGE_CHUNK_INDEX,
    PAGE_METHOD,
    Evidence,
    EvidenceLedger,
)

pytestmark = pytest.mark.unit


class StubDocument:
    def __init__(self, label="A", document_id="doc-1", title="Paper", filename="a.pdf"):
        self.label = label
        self.document_id = document_id
        self.title = title
        self.filename = filename


def _source(content, **overrides):
    base = {"content": content, "chunk_index": 0, "page": 3}
    base.update(overrides)
    return base


class TestEvidence:
    def test_excerpt_bounded(self):
        evidence = Evidence(
            evidence_id="E1",
            label="A",
            document_id="d",
            title="t",
            document="a.pdf",
            chunk_index=0,
            page=1,
            rank=1,
            position=1,
            method="dense",
            content="x" * (EXCERPT_CHARS + 100),
        )
        assert len(evidence.excerpt) == EXCERPT_CHARS
        assert evidence.shown == evidence.excerpt

    def test_shown_full_once_read(self):
        ledger = EvidenceLedger()
        (item,) = ledger.admit(
            [_source("y" * (EXCERPT_CHARS + 10))],
            document=StubDocument(),
            method="dense",
        )
        assert item.shown == item.excerpt
        ledger.read(item.evidence_id)
        assert item.shown == item.content
        assert item.read is True

    def test_to_dict_keys(self):
        ledger = EvidenceLedger()
        (item,) = ledger.admit(
            [_source("hello")], document=StubDocument(), method="dense"
        )
        assert item.to_dict()["evidence_id"] == "E1"
        assert item.to_dict()["content"] == "hello"


class TestAdmit:
    def test_ids_sequential(self):
        ledger = EvidenceLedger()
        admitted = ledger.admit(
            [_source("one"), _source("two")], document=StubDocument(), method="dense"
        )
        assert [e.evidence_id for e in admitted] == ["E1", "E2"]
        assert ledger.ids() == ("E1", "E2")

    def test_blank_content_skipped(self):
        ledger = EvidenceLedger()
        assert (
            ledger.admit([_source("   ")], document=StubDocument(), method="dense")
            == []
        )
        assert ledger.ids() == ()

    def test_duplicate_content_keeps_id(self):
        ledger = EvidenceLedger()
        first = ledger.admit(
            [_source("same text")], document=StubDocument(), method="dense"
        )
        second = ledger.admit(
            [_source("same text")], document=StubDocument(), method="sparse"
        )
        assert second[0].evidence_id == first[0].evidence_id
        assert ledger.ids() == ("E1",)

    def test_same_text_different_documents_distinct(self):
        ledger = EvidenceLedger()
        ledger.admit(
            [_source("same")], document=StubDocument(document_id="d1"), method="dense"
        )
        ledger.admit(
            [_source("same")], document=StubDocument(document_id="d2"), method="dense"
        )
        assert ledger.ids() == ("E1", "E2")

    def test_rank_is_position_in_search(self):
        ledger = EvidenceLedger()
        admitted = ledger.admit(
            [_source("a"), _source("b")], document=StubDocument(), method="dense"
        )
        assert [e.rank for e in admitted] == [1, 2]

    def test_page_key_variants(self):
        ledger = EvidenceLedger()
        (by_page,) = ledger.admit(
            [_source("x", page=4)], document=StubDocument(), method="dense"
        )
        (by_page_no,) = ledger.admit(
            [{"content": "y", "chunk_index": 0, "page_no": 7}],
            document=StubDocument(document_id="d2"),
            method="dense",
        )
        assert by_page.page == 4
        assert by_page_no.page == 7

    def test_missing_hash_computed(self):
        ledger = EvidenceLedger()
        (item,) = ledger.admit(
            [{"content": "no hash given", "chunk_index": 2}],
            document=StubDocument(),
            method="dense",
        )
        assert item.chunk_index == 2
        again = ledger.admit(
            [{"content": "no hash given", "chunk_index": 9}],
            document=StubDocument(),
            method="dense",
        )
        assert again[0].evidence_id == item.evidence_id


class TestGetAndRead:
    def test_get_case_insensitive_whitespace(self):
        ledger = EvidenceLedger()
        ledger.admit([_source("x")], document=StubDocument(), method="dense")
        assert ledger.get(" e1 ") is not None
        assert ledger.get("E99") is None

    def test_read_unknown_is_none(self):
        assert EvidenceLedger().read("E1") is None

    def test_held_and_to_dict_order(self):
        ledger = EvidenceLedger()
        ledger.admit(
            [_source("a"), _source("b")], document=StubDocument(), method="dense"
        )
        assert [e.evidence_id for e in ledger.held()] == ["E1", "E2"]
        assert [d["evidence_id"] for d in ledger.to_dict()] == ["E1", "E2"]


class TestAdmitPage:
    def test_page_evidence_marked(self):
        ledger = EvidenceLedger()
        item = ledger.admit_page(
            document=StubDocument(), page=5, content="full page text"
        )
        assert item.page == 5
        assert item.chunk_index == PAGE_CHUNK_INDEX
        assert item.method == PAGE_METHOD
        assert item.read is True
        assert item.evidence_id.startswith(EVIDENCE_PREFIX)

    def test_same_page_twice_one_id(self):
        ledger = EvidenceLedger()
        first = ledger.admit_page(document=StubDocument(), page=5, content="text")
        second = ledger.admit_page(document=StubDocument(), page=5, content="text")
        assert first.evidence_id == second.evidence_id

    def test_changed_page_text_new_id(self):
        ledger = EvidenceLedger()
        first = ledger.admit_page(document=StubDocument(), page=5, content="v1")
        second = ledger.admit_page(document=StubDocument(), page=5, content="v2")
        assert first.evidence_id != second.evidence_id
