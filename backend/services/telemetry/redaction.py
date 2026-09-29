"""
What a trace is allowed to say.

A trace exists so an operator can explain an answer: which index generation was
queried, which Passages were candidates, what the model was asked for, how long
it took, what it cost. None of that needs the words of the question, the answer,
the prompt, or the Document, and all of it would be a second copy of private
material living somewhere with weaker protection than the store itself. So the
rule here is simple and enforced in one place: a value whose name says it holds
private text is replaced by a fingerprint of that text, never by the text.

What survives is what a trace is for — identifiers, ranks, scores, counts,
lengths, durations, model names, and the hash of a Passage — plus the
credentials rule, which is not a default but a hard one. A key or a token is
stripped from any value, including values captured under the opt-in below, so a
debugging session can never widen the leak to the machine's secrets.
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import Any

#: What a redacted value is replaced with on screen and in a trace file.
REDACTED = "[redacted]"

#: Attribute names that hold a question, an answer, a prompt, or Document text.
#: A value under any of these names is fingerprinted, and is only ever written
#: out when a local capture is explicitly enabled and still inside its window.
TEXT_ATTRIBUTES = frozenset(
    {
        "answer",
        "claim",
        "claims",
        "content",
        "context",
        "expanded_query",
        "prior_questions",
        "prior_turns",
        "prompt",
        "question",
        "query",
        "rewritten_query",
        "source_text",
        "system_instruction",
        "text",
        "transcript",
        "user_prompt",
    }
)

#: Attribute names that hold a secret. These are removed whatever else is
#: allowed, so no capture mode can put a credential in a trace.
CREDENTIAL_ATTRIBUTES = frozenset(
    {
        "api_key",
        "app_secret",
        "authorization",
        "ciphertext",
        "credentials",
        "encrypted_api_key",
        "password",
        "secret",
        "token",
    }
)

#: How much of a credential is recognized in free text. Provider key formats
#: are matched on their distinctive prefixes rather than on a shape, so a long
#: document identifier is not mistaken for a key.
_CREDENTIAL_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"gsk_[A-Za-z0-9_-]{8,}"),
    re.compile(r"AIza[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._-]{8,}"),
)

#: How much of a fingerprint is kept. Enough to tell two answers apart in a
#: list, short enough that the fingerprint is not the text it stands for.
_FINGERPRINT_CHARS = 16


def credential_free(value: str) -> str:
    """Return the value with anything shaped like a credential removed."""
    text = str(value)
    for pattern in _CREDENTIAL_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def fingerprint(value: str | None) -> dict[str, Any]:
    """
    Describe private text by what can be said about it without repeating it.

    Two spans about the same answer share a fingerprint, so a trace still shows
    that both were about the same question; the text itself is not recoverable
    from a truncated digest.
    """
    text = "" if value is None else str(value)
    return {
        "redacted": REDACTED,
        "chars": len(text),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[
            :_FINGERPRINT_CHARS
        ],
    }


class Redactor:
    """
    Decides what one trace may record, and for how long it may record it.

    ``capture_text`` is the opt-in that writes questions, answers, and Document
    text into a local trace instead of a fingerprint. It is deliberately not a
    boolean that stays true for the process: the window starts when the
    redactor is built and closes after ``capture_window_seconds``, so a
    debugging session cannot silently become the way the application runs.
    Credentials are stripped from everything either way.
    """

    def __init__(
        self,
        *,
        capture_text: bool = False,
        capture_window_seconds: float = 900.0,
        clock=time.time,
    ) -> None:
        """Open the capture window, if one was asked for at all."""
        self._capture_text = bool(capture_text)
        self._window_seconds = max(0.0, float(capture_window_seconds))
        self._clock = clock
        self._deadline = self._clock() + self._window_seconds

    @property
    def capturing(self) -> bool:
        """Report whether private text is still being written out."""
        return self._capture_text and self._clock() < self._deadline

    def scrub(self, attributes: dict[str, Any]) -> dict[str, Any]:
        """Return the attributes a trace may record."""
        return self._scrub_mapping(attributes)

    def _scrub_mapping(self, attributes: dict[str, Any]) -> dict[str, Any]:
        scrubbed: dict[str, Any] = {}
        for key, value in attributes.items():
            name = str(key)
            if name in CREDENTIAL_ATTRIBUTES:
                scrubbed[name] = REDACTED
            else:
                scrubbed[name] = self._scrub_value(name, value)
        return scrubbed

    def _scrub_value(self, name: str, value: Any) -> Any:
        """
        Return one value as this name allows it to be recorded.

        A name that holds private text is fingerprinted for the whole subtree
        below it, so a Passage's text is redacted whether it was recorded as a
        string, as a list of sources, or as a mapping of them.
        """
        if name in TEXT_ATTRIBUTES and not self.capturing:
            return fingerprint(value if isinstance(value, str) else _text_of(value))
        return self._scrub_tree(value)

    def _scrub_tree(self, value: Any) -> Any:
        """Walk a value of any shape, keeping identifiers and removing secrets."""
        if isinstance(value, str):
            return credential_free(value)
        if isinstance(value, dict):
            return self._scrub_mapping(value)
        if isinstance(value, (list, tuple)):
            return [self._scrub_tree(item) for item in value]
        return value


def _text_of(value: Any) -> str:
    """Return the text inside a value of any shape, for fingerprinting it."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(str(_text_of(item)) for item in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_text_of(item) for item in value)
    return str(value)
