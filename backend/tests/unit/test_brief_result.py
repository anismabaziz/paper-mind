"""Unit tests for structured brief parsing and validation.

Pure JSON handling. Run with plain ``pytest``.
"""

import json

import pytest

from services.brief.result import (
    BRIEF_CLOSE,
    BRIEF_OPEN,
    BriefClaim,
    StructuredBrief,
    brief_from_answer,
    parse_structured_brief,
    validate_structured_brief,
)

pytestmark = pytest.mark.unit


def _block(payload):
    return f"Summary line.\n{BRIEF_OPEN}\n{json.dumps(payload)}\n{BRIEF_CLOSE}"


def _payload(**overrides):
    base = {
        "summary": "Both papers agree on X.",
        "claims": [
            {"claim": "X holds", "supports": ["E1"], "conflicts": []},
            {"claim": "Y disputed", "supports": ["E1"], "conflicts": ["E2"]},
        ],
        "gaps": ["Whether Z generalizes."],
        "abstained": False,
    }
    base.update(overrides)
    return base


class TestParseStructuredBrief:
    def test_valid_block(self):
        parsed = parse_structured_brief(_block(_payload()))
        assert parsed is not None
        assert parsed.summary == "Both papers agree on X."
        assert len(parsed.claims) == 2
        assert parsed.claims[0].supports == ("E1",)
        assert parsed.claims[1].status == "contested"
        assert parsed.gaps == ("Whether Z generalizes.",)
        assert parsed.abstained is False

    def test_no_block_is_none(self):
        assert parse_structured_brief("just prose") is None

    def test_bare_json_accepted(self):
        parsed = parse_structured_brief(json.dumps(_payload()))
        assert parsed is not None
        assert parsed.summary.startswith("Both papers")

    def test_non_json_block_keeps_summary(self):
        parsed = parse_structured_brief(
            f"Reader summary.\n{BRIEF_OPEN}\nnot json\n{BRIEF_CLOSE}"
        )
        assert parsed is not None
        assert parsed.summary == "Reader summary."
        assert parsed.claims == ()

    def test_block_without_summary_falls_back_to_prose(self):
        parsed = parse_structured_brief(_block({"claims": []}))
        assert parsed is not None
        assert parsed.summary == "Summary line."
        assert parsed.claims == ()

    def test_bad_claim_lines_dropped(self):
        payload = _payload(
            claims=["nope", {"nope": 1}, {"claim": "  ", "supports": []}]
        )
        parsed = parse_structured_brief(_block(payload))
        assert parsed is not None
        assert parsed.claims == ()

    def test_status_derived_when_missing(self):
        payload = _payload(
            claims=[
                {"claim": "a", "supports": [], "conflicts": []},
                {"claim": "b", "supports": ["E1"], "conflicts": []},
                {"claim": "c", "supports": [], "conflicts": ["E2"]},
            ]
        )
        parsed = parse_structured_brief(_block(payload))
        assert [c.status for c in parsed.claims] == [
            "unresolved",
            "supported",
            "contested",
        ]

    def test_explicit_status_kept(self):
        payload = _payload(
            claims=[{"claim": "a", "supports": ["E1"], "status": "supported"}]
        )
        parsed = parse_structured_brief(_block(payload))
        assert parsed.claims[0].status == "supported"

    def test_singular_key_aliases(self):
        payload = _payload(
            claims=[{"claim": "a", "support": ["e1"], "conflict": ["e2"]}]
        )
        parsed = parse_structured_brief(_block(payload))
        assert parsed.claims[0].supports == ("E1",)
        assert parsed.claims[0].conflicts == ("E2",)

    def test_gaps_accept_dict_shape(self):
        payload = _payload(gaps=[{"gap": "  open question  "}, 5, ""])
        parsed = parse_structured_brief(_block(payload))
        assert parsed.gaps == ("open question",)

    def test_summary_truncated(self):
        parsed = parse_structured_brief(_block(_payload(summary="s" * 5000)))
        assert len(parsed.summary) == 2000

    def test_to_dict(self):
        parsed = parse_structured_brief(_block(_payload()))
        payload = parsed.to_dict()
        assert payload["claims"][0]["order"] == 1
        assert payload["abstained"] is False


class TestValidateStructuredBrief:
    def _brief(self):
        return StructuredBrief(
            summary="s",
            claims=(
                BriefClaim(1, "a", ("E1", "E9"), (), "supported"),
                BriefClaim(2, "b", (), (), "unresolved"),
            ),
            gaps=(),
            abstained=False,
        )

    def test_invalid_ids_dropped_and_reported(self):
        validated, invalid = validate_structured_brief(self._brief(), ["E1"])
        assert validated.claims[0].supports == ("E1",)
        assert invalid == ("E9",)
        assert validated.claims[0].status == "supported"

    def test_emptied_claim_becomes_unresolved(self):
        brief = StructuredBrief(
            summary="s",
            claims=(BriefClaim(1, "a", ("E9",), (), "supported"),),
            gaps=(),
            abstained=False,
        )
        validated, _ = validate_structured_brief(brief, ["E1"])
        assert validated.claims[0].status == "unresolved"

    def test_explicit_unresolved_stays(self):
        brief = StructuredBrief(
            summary="s",
            claims=(BriefClaim(1, "a", ("E1",), (), "unresolved"),),
            gaps=(),
            abstained=False,
        )
        validated, _ = validate_structured_brief(brief, ["E1"])
        assert validated.claims[0].status == "unresolved"

    def test_invalid_ids_deduped_in_order(self):
        brief = StructuredBrief(
            summary="s",
            claims=(
                BriefClaim(1, "a", ("E9",), (), "supported"),
                BriefClaim(2, "b", (), ("E9", "E8"), "supported"),
            ),
            gaps=(),
            abstained=False,
        )
        _, invalid = validate_structured_brief(brief, ["E1"])
        assert invalid == ("E9", "E8")


class TestBriefFromAnswer:
    def test_block_validated(self):
        brief, invalid = brief_from_answer(_block(_payload()), ["E1", "E2"])
        assert brief.summary.startswith("Both papers")
        assert invalid == ()

    def test_prose_becomes_summary(self):
        brief, invalid = brief_from_answer("plain summary", ["E1"])
        assert brief.summary == "plain summary"
        assert brief.claims == ()
        assert invalid == ()

    def test_empty_answer(self):
        brief, _ = brief_from_answer("", ["E1"])
        assert brief.summary == ""
