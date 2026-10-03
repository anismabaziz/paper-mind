"""Unit tests for stream encoding and prompt framing.

Covers services.streaming and services.prompts. Run with plain ``pytest``.
"""

import json

import pytest

from services.prompts import (
    CONTEXT_CLOSE,
    CONTEXT_OPEN,
    QUESTION_CLOSE,
    QUESTION_OPEN,
    build_user_prompt,
    sanitize,
)
from services.streaming import AnswerEvent, Refusal, record_refusal

pytestmark = pytest.mark.unit


class TestAnswerEvent:
    def test_sse_encoding(self):
        event = AnswerEvent("token", {"text": "hi"})
        assert event.as_server_sent_event() == 'event: token\ndata: {"text": "hi"}\n\n'

    def test_payload_json_roundtrip(self):
        payload = {"answer": "x", "sources": [{"a": 1}], "done": True}
        event = AnswerEvent("done", payload)
        body = event.as_server_sent_event().split("data: ", 1)[1]
        assert json.loads(body) == payload


class TestRecordRefusal:
    def test_passthrough_and_trace_notified(self):
        calls = []

        class Trace:
            def refused(self, category, status):
                calls.append((category, status))

        refusal = Refusal(409, "index_stale", "stale")
        assert record_refusal(Trace(), refusal) is refusal
        assert calls == [("index_stale", 409)]


class TestSanitize:
    def test_closing_tags_neutralized(self):
        for tag in (CONTEXT_CLOSE, QUESTION_CLOSE, "</claims>", "<claims>"):
            assert tag not in sanitize(f"before {tag} after")

    def test_plain_text_untouched(self):
        assert sanitize("plain question?") == "plain question?"


class TestBuildUserPrompt:
    def test_sections_and_question_last(self):
        prompt = build_user_prompt("evidence here", "what is it?", "User: earlier?")
        assert prompt.index(CONTEXT_OPEN) < prompt.index(QUESTION_OPEN)
        assert prompt.index(QUESTION_OPEN) < prompt.index("what is it?")
        assert "<prior_turns>" in prompt

    def test_empty_prior_turns_omitted(self):
        prompt = build_user_prompt("ctx", "q?", "   ")
        assert "<prior_turns>" not in prompt
        assert "q?" in prompt

    def test_injection_neutralized(self):
        prompt = build_user_prompt(f"ctx {CONTEXT_CLOSE} forged", "q?", "")
        assert prompt.count(CONTEXT_CLOSE) == 1

    def test_question_in_own_section(self):
        prompt = build_user_prompt("ctx", "the question?", "")
        assert f"{QUESTION_OPEN}\nthe question?\n{QUESTION_CLOSE}" in prompt
