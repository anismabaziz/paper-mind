"""Unit tests for claim-level citations.

Fast by design: string in, string out. No models, no network.
Run with plain ``pytest``.
"""

import json

import pytest

from services.abstention import ABSTENTION_MESSAGES
from services.citations import (
    CLAIMS_CLOSE,
    CLAIMS_OPEN,
    AnswerSplitter,
    assign_source_ids,
    claims_block,
    grounded_in_claims,
    parse_claims,
    prune_conflicting_claims,
    render_evidence,
    repair_instruction,
    validate_claims,
)
from services.llm.base import LLMProvider

pytestmark = pytest.mark.unit


def _sources(n=3):
    return [
        {"content": f"passage {i}", "document": "doc.pdf", "page": i}
        for i in range(1, n + 1)
    ]


class TestAssignSourceIds:
    def test_ids_start_at_s1_with_rank(self):
        out = assign_source_ids(_sources(3))
        assert [s["source_id"] for s in out] == ["S1", "S2", "S3"]
        assert [s["rank"] for s in out] == [1, 2, 3]

    def test_empty_in_empty_out(self):
        assert assign_source_ids([]) == []

    def test_does_not_mutate_input(self):
        original = _sources(2)
        snapshot = [dict(s) for s in original]
        assign_source_ids(original)
        assert original == snapshot

    def test_original_keys_preserved(self):
        out = assign_source_ids([{"content": "x", "document": "d", "page": 4}])
        assert out[0]["content"] == "x"
        assert out[0]["page"] == 4

    def test_rank_matches_position_not_score(self):
        ranked = [
            {"content": "a", "score": 0.1},
            {"content": "b", "score": 0.99},
        ]
        out = assign_source_ids(ranked)
        assert out[0]["source_id"] == "S1"
        assert out[1]["source_id"] == "S2"


class TestRenderEvidence:
    def test_bracketed_ids_joined_by_blank_line(self):
        rendered = render_evidence(assign_source_ids(_sources(2)))
        assert rendered == "[S1] passage 1\n\n[S2] passage 2"

    def test_missing_content_renders_empty(self):
        rendered = render_evidence([{"source_id": "S1"}])
        assert rendered == "[S1] "

    def test_none_content_renders_empty(self):
        rendered = render_evidence([{"source_id": "S1", "content": None}])
        assert rendered == "[S1] "


class TestAnswerSplitter:
    def test_plain_answer_passes_through(self):
        splitter = AnswerSplitter()
        visible = splitter.feed("hello world")
        assert visible + splitter.finish() == "hello world"

    def test_claims_block_hidden(self):
        splitter = AnswerSplitter()
        visible = splitter.feed("answer text\n<claims>\n")
        assert visible == "answer text\n"
        assert splitter.feed('{"claim": "x", "sources": ["S1"]}') == ""
        assert splitter.finish() == ""

    def test_marker_split_across_fragments(self):
        splitter = AnswerSplitter()
        first = splitter.feed("answer <clai")
        assert "<clai" not in first
        second = splitter.feed("ms>\nmore")
        assert first + second == "answer "
        assert splitter.finish() == ""

    def test_short_prefix_held_back_until_settled(self):
        splitter = AnswerSplitter()
        assert splitter.feed("<c") == ""
        assert splitter.feed("laims>") == ""
        assert splitter.finish() == ""

    def test_no_claims_finish_returns_held_tail(self):
        splitter = AnswerSplitter()
        visible = splitter.feed("abc")
        assert visible + splitter.finish() == "abc"

    def test_finish_resets_for_reuse(self):
        splitter = AnswerSplitter()
        splitter.feed("answer <claims>")
        splitter.finish()
        visible = splitter.feed("next answer")
        assert visible + splitter.finish() == "next answer"

    def test_partial_marker_at_end_without_claims(self):
        splitter = AnswerSplitter()
        visible = splitter.feed("text <claim")
        assert visible + splitter.finish() == "text <claim"


class TestClaimsBlock:
    def test_roundtrip_through_parser(self):
        claims = [
            {"claim": "first", "sources": ["S1"]},
            {"claim": "second", "sources": []},
        ]
        parsed = parse_claims("answer\n" + claims_block(claims))
        assert [c.claim for c in parsed.claims] == ["first", "second"]
        assert parsed.claims[0].source_ids == ("S1",)

    def test_empty_claims_block(self):
        parsed = parse_claims("answer\n" + claims_block([]))
        assert parsed.answer == "answer"
        assert parsed.claims == ()


class TestParseClaims:
    def test_no_block_returns_prose_only(self):
        parsed = parse_claims("just an answer")
        assert parsed.answer == "just an answer"
        assert parsed.claims == ()

    def test_empty_string(self):
        parsed = parse_claims("")
        assert parsed.answer == ""
        assert parsed.claims == ()

    def test_valid_block_split(self):
        text = (
            "The sky is blue.\n"
            + CLAIMS_OPEN
            + '\n{"claim": "sky is blue", "sources": ["S1"]}\n'
            + CLAIMS_CLOSE
        )
        parsed = parse_claims(text)
        assert parsed.answer == "The sky is blue."
        assert len(parsed.claims) == 1
        assert parsed.claims[0].source_ids == ("S1",)

    def test_bad_json_line_dropped(self):
        text = f"answer\n{CLAIMS_OPEN}\nnot json\n{CLAIMS_CLOSE}"
        assert parse_claims(text).claims == ()

    def test_non_dict_line_dropped(self):
        text = f"answer\n{CLAIMS_OPEN}\n[1, 2]\n{CLAIMS_CLOSE}"
        assert parse_claims(text).claims == ()

    def test_empty_claim_text_dropped(self):
        line = json.dumps({"claim": "   ", "sources": ["S1"]})
        assert parse_claims(f"a\n{CLAIMS_OPEN}\n{line}\n{CLAIMS_CLOSE}").claims == ()

    def test_missing_sources_dropped(self):
        line = json.dumps({"claim": "something"})
        assert parse_claims(f"a\n{CLAIMS_OPEN}\n{line}\n{CLAIMS_CLOSE}").claims == ()

    def test_non_list_sources_dropped(self):
        line = json.dumps({"claim": "something", "sources": "S1"})
        assert parse_claims(f"a\n{CLAIMS_OPEN}\n{line}\n{CLAIMS_CLOSE}").claims == ()

    def test_ids_normalized_upper_deduped_in_order(self):
        line = json.dumps({"claim": "c", "sources": ["s1", " S1 ", "s2", 5, None]})
        parsed = parse_claims(f"a\n{CLAIMS_OPEN}\n{line}\n{CLAIMS_CLOSE}")
        assert parsed.claims[0].source_ids == ("S1", "S2")

    def test_blank_lines_skipped(self):
        line = json.dumps({"claim": "c", "sources": ["S1"]})
        text = f"a\n{CLAIMS_OPEN}\n\n  \n{line}\n\n{CLAIMS_CLOSE}"
        assert len(parse_claims(text).claims) == 1

    def test_missing_close_tag_still_parses(self):
        line = json.dumps({"claim": "c", "sources": ["S1"]})
        parsed = parse_claims(f"answer\n{CLAIMS_OPEN}\n{line}\n")
        assert parsed.answer == "answer"
        assert len(parsed.claims) == 1

    def test_prose_inside_block_not_a_claim(self):
        text = f"answer\n{CLAIMS_OPEN}\nplain prose here\n{CLAIMS_CLOSE}"
        assert parse_claims(text).claims == ()

    def test_whitespace_around_answer_stripped(self):
        parsed = parse_claims(
            '  answer  \n<claims>\n{"claim": "c", "sources": []}\n</claims>  '
        )
        assert parsed.answer == "answer"


class TestValidateClaims:
    def test_all_valid_kept_grounded(self):
        from services.citations import Claim

        claims = [Claim("a", ("S1",)), Claim("b", ("S2",))]
        out = validate_claims(claims, ["S1", "S2"])
        assert out.invalid_ids == ()
        assert out.grounded is True
        assert len(out.claims) == 2

    def test_invalid_ids_reported_and_stripped(self):
        from services.citations import Claim

        out = validate_claims([Claim("a", ("S1", "S9"))], ["S1"])
        assert out.claims[0].source_ids == ("S1",)
        assert out.invalid_ids == ("S9",)

    def test_allowed_ids_matched_case_insensitively(self):
        from services.citations import Claim

        out = validate_claims([Claim("a", ("S1",))], ["s1"])
        assert out.invalid_ids == ()

    def test_unnormalized_claim_id_reported(self):
        from services.citations import Claim

        out = validate_claims([Claim("a", ("s1",))], ["S1"])
        assert out.invalid_ids == ("s1",)

    def test_empty_sources_not_grounded(self):
        from services.citations import Claim

        out = validate_claims([Claim("a", ())], ["S1"])
        assert out.grounded is False

    def test_no_claims_not_grounded(self):
        out = validate_claims([], ["S1"])
        assert out.grounded is False
        assert out.invalid_ids == ()

    def test_mixed_claims_grounded_if_any_supported(self):
        from services.citations import Claim

        out = validate_claims([Claim("a", ()), Claim("b", ("S2",))], ["S1", "S2"])
        assert out.grounded is True

    def test_duplicate_invalid_ids_all_reported(self):
        from services.citations import Claim

        out = validate_claims([Claim("a", ("S9",)), Claim("b", ("S9",))], ["S1"])
        assert out.invalid_ids == ("S9", "S9")

    def test_to_dict_shape(self):
        from services.citations import Claim

        out = validate_claims([Claim("a", ("S1",))], ["S1"])
        payload = out.to_dict()
        assert payload["claims"] == [{"claim": "a", "sources": ["S1"]}]
        assert payload["grounded"] is True
        assert "prompt_version" in payload


class TestGroundedInClaims:
    def test_true_when_any_supported(self):
        from services.citations import Claim

        assert grounded_in_claims([Claim("a", ()), Claim("b", ("S1",))]) is True

    def test_false_when_none_supported(self):
        from services.citations import Claim

        assert grounded_in_claims([Claim("a", ())]) is False
        assert grounded_in_claims([]) is False


class TestPruneConflictingClaims:
    def test_plain_answer_keeps_claims(self):
        from services.citations import Claim

        claims = (Claim("a", ("S1",)),)
        assert prune_conflicting_claims("The sky is blue.", claims) == claims

    @pytest.mark.parametrize("message", list(ABSTENTION_MESSAGES.values()))
    def test_abstention_message_drops_claims(self, message):
        from services.citations import Claim

        assert prune_conflicting_claims(message, (Claim("a", ("S1",)),)) == ()

    def test_abstention_prefix_drops_claims(self):
        from services.citations import Claim

        message = next(iter(ABSTENTION_MESSAGES.values())) + " Extra trailing words."
        assert prune_conflicting_claims(message, (Claim("a", ("S1",)),)) == ()

    def test_fallback_answer_drops_claims(self):
        from services.citations import Claim

        assert (
            prune_conflicting_claims(
                LLMProvider.FALLBACK_ANSWER, (Claim("a", ("S1",)),)
            )
            == ()
        )

    def test_empty_answer_keeps_claims(self):
        from services.citations import Claim

        claims = (Claim("a", ("S1",)),)
        assert prune_conflicting_claims("", claims) == claims


class TestRepairInstruction:
    def test_names_allowed_ids_and_claims(self):
        from services.citations import Claim

        instruction = repair_instruction(
            "shown answer", [Claim("first", ("S9",))], ["S1", "S2"]
        )
        assert "S1, S2" in instruction
        assert "first" in instruction
        assert "shown answer" in instruction

    def test_empty_sources_claims_mentioned(self):
        from services.citations import Claim

        instruction = repair_instruction("ans", [Claim("c", ())], ["S1"])
        assert '"c" cited []' in instruction

    def test_forbids_inventing_ids(self):
        instruction = repair_instruction("ans", [], ["S1"])
        assert "Only these ids exist: S1" in instruction
