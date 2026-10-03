"""Unit tests for deterministic query expansion.

No model calls. The rewriter is injected. Run with plain ``pytest``.
"""

import pytest

from services.retrieval.query_expansion import (
    QueryExpansion,
    expand_query,
    refers_to_prior_turns,
)

pytestmark = pytest.mark.unit


class TestRefersToPriorTurns:
    @pytest.mark.parametrize(
        "query",
        ["what is it about?", "tell me about that method", "these results confuse me"],
    )
    def test_pronouns_refer_back(self, query):
        assert refers_to_prior_turns(query) is True

    @pytest.mark.parametrize(
        "query",
        [
            "what about the second method?",
            "the last chapter summary",
            "any other approaches?",
        ],
    )
    def test_ordinals_refer_back(self, query):
        assert refers_to_prior_turns(query) is True

    def test_reference_phrase(self):
        assert refers_to_prior_turns("as you said earlier, what Lily paper?") is True
        assert refers_to_prior_turns("you mentioned a dataset, which one?") is True

    @pytest.mark.parametrize(
        "query",
        [
            "what is photosynthesis?",
            "summarize chapter three",
            "run the training again",
        ],
    )
    def test_standalone_questions_do_not(self, query):
        assert refers_to_prior_turns(query) is False

    def test_empty_query(self):
        assert refers_to_prior_turns("") is False


class TestExpandQuery:
    def test_no_prior_returns_unchanged(self):
        out = expand_query("what is it?", [], max_chars=2000)
        assert out == QueryExpansion("what is it?", "what is it?", "none")

    def test_standalone_with_prior_returns_unchanged(self):
        out = expand_query("what is photosynthesis?", ["what is DNA?"], max_chars=2000)
        assert out.expanded_query == "what is photosynthesis?"
        assert out.method == "none"

    def test_followup_prepends_prior(self):
        out = expand_query("what is it?", ["what is photosynthesis?"], max_chars=2000)
        assert out.method == "deterministic"
        assert out.expanded_query == "what is photosynthesis? what is it?"

    def test_blank_prior_questions_skipped(self):
        out = expand_query("what is it?", ["  ", "", "what is DNA?"], max_chars=2000)
        assert out.expanded_query == "what is DNA? what is it?"

    def test_truncation_keeps_current_question_whole(self):
        out = expand_query("what is it?", ["what is photosynthesis?"], max_chars=20)
        assert out.expanded_query.endswith("what is it?")
        assert len(out.expanded_query) <= 20 + len("what is photosynthesis?")

    def test_overlong_question_goes_whole(self):
        query = "what is it " * 100
        out = expand_query(query, ["prior?"], max_chars=10)
        assert out.expanded_query == query
        assert out.method == "deterministic"

    def test_zero_budget_returns_question(self):
        out = expand_query("what is it?", ["prior?"], max_chars=0)
        assert out.expanded_query == "what is it?"

    def test_model_rewrite_wins(self):
        out = expand_query(
            "what is it?",
            ["what is photosynthesis?"],
            max_chars=2000,
            rewrite=lambda q, p: "photosynthesis definition",
        )
        assert out.expanded_query == "photosynthesis definition"
        assert out.method == "model"

    def test_model_rewrite_truncated_to_budget(self):
        out = expand_query(
            "what is it?",
            ["prior?"],
            max_chars=10,
            rewrite=lambda q, p: "x" * 100,
        )
        assert out.expanded_query == "x" * 10

    def test_failed_rewrite_falls_back(self):
        def boom(query, prior):
            raise RuntimeError("provider down")

        out = expand_query(
            "what is it?", ["what is DNA?"], max_chars=2000, rewrite=boom
        )
        assert out.method == "deterministic"
        assert "what is DNA?" in out.expanded_query

    def test_empty_rewrite_falls_back(self):
        out = expand_query(
            "what is it?", ["what is DNA?"], max_chars=2000, rewrite=lambda q, p: "   "
        )
        assert out.method == "deterministic"

    def test_non_string_rewrite_falls_back(self):
        out = expand_query(
            "what is it?", ["what is DNA?"], max_chars=2000, rewrite=lambda q, p: None
        )
        assert out.method == "deterministic"

    def test_to_dict_shape(self):
        out = expand_query("q?", [], max_chars=10)
        assert out.to_dict() == {
            "original_query": "q?",
            "expanded_query": "q?",
            "query_expansion": "none",
        }
