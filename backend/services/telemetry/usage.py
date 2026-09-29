"""
What one model call read and what it wrote.

The providers this app talks to stream text without reporting usage, so the
counts here are the app's own tokenizer over the exact text it assembled. That
is an estimate, and it is labelled as one everywhere it is recorded: it is
close enough to price a change to the prompt, and it is the same measure the
evaluation run reports, so a trace and a report can be compared.
"""

from __future__ import annotations

from services.chat_context import token_count
from services.prompts import SYSTEM_INSTRUCTION


def input_tokens(query: str, context: str, prior_turns: str = "") -> int:
    """
    Return what one call asks the model to read.

    The system instruction goes to the provider on every call, so leaving it
    out would understate every answer by the same amount.
    """
    return (
        token_count(SYSTEM_INSTRUCTION)
        + token_count(query)
        + token_count(context)
        + token_count(prior_turns)
    )


def output_tokens(generated: str) -> int:
    """Return what one call asked the model to write."""
    return token_count(generated)
