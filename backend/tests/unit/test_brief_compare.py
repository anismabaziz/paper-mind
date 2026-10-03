"""Unit tests for deterministic evidence comparison.

Pure term arithmetic. Run with plain ``pytest``.
"""

import pytest

from services.brief.compare import QUOTED_CHARS, compare_items, significant_terms
from services.brief.evidence import EvidenceLedger

pytestmark = pytest.mark.unit


class StubDocument:
    def __init__(self, label, document_id=None):
        self.label = label
        self.document_id = document_id or f"doc-{label}"
        self.title = f"Paper {label}"
        self.filename = f"{label}.pdf"


def _held(texts_by_label):
    ledger = EvidenceLedger()
    for label, texts in texts_by_label.items():
        for text in texts:
            ledger.admit(
                [{"content": text, "chunk_index": 0}],
                document=StubDocument(label),
                method="dense",
            )
    return ledger.held()


class TestSignificantTerms:
    def test_drops_short_and_stopwords(self):
        terms = significant_terms("the cat is an absolute triumph of engineering")
        assert "the" not in terms
        assert "cat" in terms
        assert "engineering" in terms
        assert all(len(t) >= 3 for t in terms)

    def test_first_seen_order_no_repeats(self):
        terms = significant_terms("alpha beta alpha gamma beta")
        assert terms == ("alpha", "beta", "gamma")

    def test_empty(self):
        assert significant_terms("") == ()


class TestCompareItems:
    OVERLAP = (
        "photosynthesis chlorophyll chloroplast thylakoid sunlight carbon fixation "
        "calvin cycle electron transport chain reaction center"
    )

    def test_empty(self):
        out = compare_items([])
        assert out["compared"] == []
        assert out["similarities"] == []
        assert out["differences"] == []
        assert out["contradictions"] == []
        assert "surface comparison" in out["note"]

    def test_similar_pair_shares_terms(self):
        held = _held({"A": [self.OVERLAP + " alpha"], "B": [self.OVERLAP + " beta"]})
        out = compare_items(held)
        assert len(out["similarities"]) == 1
        entry = out["similarities"][0]
        assert set(entry["evidence_ids"]) == {held[0].evidence_id, held[1].evidence_id}
        assert len(entry["shared_terms"]) >= 1

    def test_unrelated_pair_no_similarity(self):
        held = _held(
            {
                "A": ["quantum entanglement photon polarization"],
                "B": ["medieval agrarian tax policy"],
            }
        )
        assert compare_items(held)["similarities"] == []

    def test_differences_name_provenance_and_distinctive_terms(self):
        held = _held(
            {
                "A": ["photosynthesis chlorophyll sunlight"],
                "B": ["photosynthesis mitochondria respiration"],
            }
        )
        differences = compare_items(held)["differences"]
        assert differences
        assert "A" in differences[0]["detail"] and "B" in differences[0]["detail"]
        assert any("chlorophyll" in entry["detail"] for entry in differences[1:])

    def test_contrast_cue_flags_candidate(self):
        shared = self.OVERLAP
        held = _held(
            {
                "A": [shared + " effective"],
                "B": [shared + " however limited insufficient"],
            }
        )
        contradictions = compare_items(held)["contradictions"]
        assert len(contradictions) == 1
        assert "Candidate contrast" in contradictions[0]["detail"]

    def test_same_label_never_contradiction(self):
        held = _held({"A": [self.OVERLAP + " however limited"], "A2": [self.OVERLAP]})
        for item in held:
            item.label = "A"
        assert compare_items(held)["contradictions"] == []

    def test_long_text_truncated_in_quote(self):
        held = _held({"A": ["x " * (QUOTED_CHARS + 100)]})
        (quoted,) = compare_items(held)["compared"]
        assert quoted["truncated"] is True
        assert len(quoted["text"]) == QUOTED_CHARS

    def test_entries_only_name_given_ids(self):
        held = _held({"A": [self.OVERLAP], "B": [self.OVERLAP + " however limited"]})
        known = {item.evidence_id for item in held}
        out = compare_items(held)
        for group in ("similarities", "differences", "contradictions"):
            for entry in out[group]:
                assert set(entry["evidence_ids"]) <= known
