"""How this codebase splits and measures text: words and tokens."""

import re

import tiktoken

#: Words too common to carry meaning in retrieval and grading. One set serves
#: both so a Passage that retrieval ignores is a Passage grading ignores too.
#: Topic-term extraction keeps its own set: contrast cues must stay visible
#: there, so it is deliberately not this one.
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "where",
        "which",
        "with",
    }
)

_WORD_RE = re.compile(r"[a-z0-9%°]+")


def word_parts(text: str) -> list[str]:
    """
    Split text into lowercase words, keeping stopwords.

    One splitter so retrieval, grading, and query expansion agree on what a
    word is; filtering decisions stay with each caller, which need different
    ones.
    """
    return _WORD_RE.findall(text.lower())


# cl100k_base is the tokenizer for chunking and context budgets alike;
# stable, no download. One instance so both agree on every count.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def token_len(text: str) -> int:
    """
    Return the token count of a piece of text.

    One instance so chunking and context budgets count the same way; two
    encodings can only ever agree by accident.
    """
    return len(_ENCODING.encode(text)) if text else 0
