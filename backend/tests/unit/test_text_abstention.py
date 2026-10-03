"""Unit tests for text splitting and answer abstention.

Pure string logic. Run with plain ``pytest``.
"""

import pytest

from services.abstention import (
    ABSTENTION_MESSAGES,
    EVIDENCE_UNUSABLE,
    NO_EVIDENCE,
    Abstention,
    abstention_for,
    is_usable,
)
from services.text import STOPWORDS, token_len, word_parts

pytestmark = pytest.mark.unit


class TestWordParts:
    def test_lowercases_and_splits(self):
        assert word_parts("Hello World") == ["hello", "world"]

    def test_keeps_stopwords(self):
        assert word_parts("the cat and the hat") == ["the", "cat", "and", "the", "hat"]

    def test_drops_punctuation(self):
        assert word_parts("well-known: results!") == ["well", "known", "results"]

    def test_keeps_numbers_and_percent(self):
        assert word_parts("100% accurate, 42nd run") == [
            "100%",
            "accurate",
            "42nd",
            "run",
        ]

    def test_empty_string(self):
        assert word_parts("") == []

    def test_stopwords_set_agrees(self):
        for word in ("the", "and", "of", "is"):
            assert word in STOPWORDS
            assert word in word_parts(f"x {word} y")


class TestTokenLen:
    def test_empty_is_zero(self):
        assert token_len("") == 0

    def test_single_word_positive(self):
        assert token_len("hello") > 0

    def test_longer_text_more_tokens(self):
        assert token_len("hello world " * 50) > token_len("hello")

    def test_deterministic(self):
        assert token_len("retrieval augmented generation") == token_len(
            "retrieval augmented generation"
        )


class TestIsUsable:
    def test_usable_source(self):
        assert is_usable({"content": "some text", "document": "a.pdf"}) is True

    def test_blank_content_unusable(self):
        assert is_usable({"content": "   ", "document": "a.pdf"}) is False

    def test_missing_content_unusable(self):
        assert is_usable({"document": "a.pdf"}) is False

    def test_non_string_content_unusable(self):
        assert is_usable({"content": 5, "document": "a.pdf"}) is False

    def test_blank_document_unusable(self):
        assert is_usable({"content": "text", "document": ""}) is False

    def test_missing_document_unusable(self):
        assert is_usable({"content": "text"}) is False


class TestAbstentionFor:
    def test_none_when_evidence_usable(self):
        sources = [{"content": "text", "document": "a.pdf"}]
        assert abstention_for(sources) is None

    def test_none_when_one_of_many_usable(self):
        sources = [
            {"content": "", "document": "a.pdf"},
            {"content": "text", "document": "a.pdf"},
        ]
        assert abstention_for(sources) is None

    def test_no_evidence_when_empty(self):
        abstention = abstention_for([])
        assert abstention is not None
        assert abstention.reason == NO_EVIDENCE
        assert abstention.retrieved == 0
        assert abstention.usable == 0
        assert abstention.message == ABSTENTION_MESSAGES[NO_EVIDENCE]

    def test_unusable_when_sources_but_none_readable(self):
        sources = [{"content": "", "document": "a.pdf"}]
        abstention = abstention_for(sources)
        assert abstention is not None
        assert abstention.reason == EVIDENCE_UNUSABLE
        assert abstention.retrieved == 1
        assert abstention.usable == 0

    def test_to_dict_shape(self):
        abstention = Abstention(reason=NO_EVIDENCE, message="m", retrieved=0, usable=0)
        assert abstention.to_dict() == {
            "abstained": True,
            "message": "m",
            "reason": NO_EVIDENCE,
            "retrieved": 0,
        }
