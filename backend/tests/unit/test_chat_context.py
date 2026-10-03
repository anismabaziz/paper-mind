"""Unit tests for bounded chat context windows.

Token budgets use the real tokenizer. Run with plain ``pytest``.
"""

import pytest

from services.chat_context import (
    bounded_sources,
    bounded_turns,
    build_chat_context,
    build_model_rewriter,
    render_prior_turns,
)
from services.llm.base import LLMProvider
from services.retrieval.query_expansion import QueryExpansion

pytestmark = pytest.mark.unit


def _turn(question, answer="an answer"):
    return {"question": question, "answer": answer}


def _source(content, rank=1):
    return {"content": content, "source_id": f"S{rank}", "rank": rank}


class TestRenderPriorTurns:
    def test_transcript_shape(self):
        rendered = render_prior_turns([_turn("q1", "a1"), _turn("q2", "a2")])
        assert rendered == "User: q1\nAssistant: a1\nUser: q2\nAssistant: a2"

    def test_blank_sides_skipped(self):
        rendered = render_prior_turns([{"question": "", "answer": "a"}])
        assert rendered == "Assistant: a"

    def test_empty(self):
        assert render_prior_turns([]) == ""


class TestBoundedTurns:
    def test_count_window_newest_first(self):
        turns = [_turn(f"q{i}") for i in range(10)]
        kept, dropped = bounded_turns(turns, max_turns=4, token_budget=10**6)
        assert [t["question"] for t in kept] == ["q6", "q7", "q8", "q9"]
        assert dropped == 6

    def test_zero_max_turns(self):
        kept, dropped = bounded_turns([_turn("q")], max_turns=0, token_budget=10**6)
        assert kept == []
        assert dropped == 1

    def test_token_budget_drops_oldest(self):
        turns = [_turn("word " * 200, "answer " * 200) for _ in range(4)]
        kept, dropped = bounded_turns(turns, max_turns=4, token_budget=50)
        assert dropped >= 1
        assert kept == turns[len(turns) - len(kept) :]

    def test_full_conversation_count_reported(self):
        turns = [_turn("q1"), _turn("q2")]
        _, dropped = bounded_turns(
            turns, max_turns=4, token_budget=10**6, turns_in_conversation=10
        )
        assert dropped == 8

    def test_nothing_dropped(self):
        kept, dropped = bounded_turns([_turn("q")], max_turns=4, token_budget=10**6)
        assert len(kept) == 1
        assert dropped == 0


class TestBoundedSources:
    def test_all_fit(self):
        sources = [_source("short text", 1), _source("more text", 2)]
        kept, dropped = bounded_sources(sources, token_budget=10**6)
        assert kept == sources
        assert dropped == 0

    def test_tail_dropped_first(self):
        sources = [_source("word " * 100, 1), _source("word " * 100, 2)]
        kept, dropped = bounded_sources(sources, token_budget=30)
        assert len(kept) == 1
        assert kept[0]["source_id"] == "S1"
        assert dropped == 1

    def test_first_source_always_kept(self):
        sources = [_source("word " * 500, 1)]
        kept, dropped = bounded_sources(sources, token_budget=1)
        assert kept == sources
        assert dropped == 0

    def test_empty(self):
        assert bounded_sources([], 100) == ([], 0)


class TestBuildChatContext:
    def _expansion(self):
        return QueryExpansion("q?", "q?", "none")

    def test_assembles_context_and_transcript(self):
        context = build_chat_context(
            "the question?",
            [_source("evidence text", 1)],
            [_turn("earlier?", "yes.")],
            self._expansion(),
            max_turns=4,
            prior_turns_token_budget=1200,
            context_token_budget=6000,
        )
        assert "[S1] evidence text" in context.context
        assert "User: earlier?" in context.prior_turns
        assert context.query == "the question?"
        assert context.dropped_turns == 0
        assert context.dropped_sources == 0

    def test_model_window_shrinks_evidence(self):
        sources = [_source("word " * 200, 1), _source("word " * 200, 2)]
        context = build_chat_context(
            "q?",
            sources,
            [],
            self._expansion(),
            max_turns=4,
            prior_turns_token_budget=1200,
            context_token_budget=6000,
            input_token_budget=300,
        )
        assert context.dropped_sources >= 1
        assert len(context.sources) < len(sources)

    def test_large_model_window_keeps_everything(self):
        sources = [_source("short", 1)]
        context = build_chat_context(
            "q?",
            sources,
            [_turn("q")],
            self._expansion(),
            max_turns=4,
            prior_turns_token_budget=1200,
            context_token_budget=6000,
            input_token_budget=10**6,
        )
        assert context.dropped_sources == 0


class TestBuildModelRewriter:
    class _Provider:
        def __init__(self, reply):
            self._reply = reply

        def generate_response(self, query, context, prior_turns=""):
            return self._reply

    def test_returns_rewrite(self):
        rewrite = build_model_rewriter(self._Provider("standalone query"))
        assert rewrite("what is it?", "prior?") == "standalone query"

    def test_fallback_answer_means_no_rewrite(self):
        rewrite = build_model_rewriter(self._Provider(LLMProvider.FALLBACK_ANSWER))
        assert rewrite("what is it?", "prior?") == ""
