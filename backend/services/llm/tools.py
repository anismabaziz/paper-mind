"""
The provider-neutral shape of one tool-using model call.

A model that decides what to read next needs three things from a provider: what
tools exist, what it said, and what it read. Those are spelled differently by
each SDK, so this module is the one vocabulary both of them translate into. The
brief loop above it never sees a Gemini ``FunctionCall`` or a Groq
``ChatCompletionMessageToolCall``.

The types are deliberately narrow. A :class:`ToolSpec` carries a name, a
description, and a JSON Schema for its arguments — nothing that could execute
anything. A :class:`ToolCall` carries an id, a name, and arguments that have
already been decoded from JSON; a call whose arguments are not a JSON object
never becomes a :class:`ToolCall`, because the brief loop validates arguments
against a schema and has nothing to validate against a string. The two reasons
this module exists are that one call can ask for several tools at once, and that
a tool call is not the end of an answer: :class:`ToolTurn` carries the prose the
model wrote alongside the calls, so a model that thinks out loud before it
searches is not truncated for it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

#: The roles a tool-using conversation is made of. A model turn carries the
#: calls it made; a tool turn carries one result per call, addressed by the call
#: id so a provider can pair them however its API requires.
MessageRole = Literal["user", "assistant", "tool"]


class ToolCallError(ValueError):
    """One tool call could not be read as a name and a JSON object of arguments."""


@dataclass(frozen=True)
class ToolSpec:
    """
    One tool the model is offered, described well enough to choose it.

    ``parameters`` is a JSON Schema object, which is what both providers already
    accept, so the schema is written once here rather than twice per provider.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    """One tool the model asked for, with its arguments already decoded."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolTurn:
    """
    What one model call produced: prose, tool calls, and what it cost.

    ``tool_calls`` empty means the model answered, and ``text`` is that answer.
    A turn with both is a model that wrote a sentence and then went looking for
    the evidence to support it, which is normal and is not truncated.

    The token counts are what the provider reported where it reports them, and
    the app's own estimate where it does not; the trace labels which it got.
    """

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class ToolMessage:
    """One entry in a tool-using conversation, in the provider-neutral shape."""

    role: MessageRole
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    #: Set on a tool entry: the id of the call this result answers.
    tool_call_id: str = ""
    #: Set on a tool entry: the tool that produced it.
    name: str = ""


def tool_call(id: str, name: str, arguments: Any) -> ToolCall:
    """
    Return one tool call, or raise when its arguments are not a JSON object.

    Both SDKs hand arguments over as a string on one provider and a dict on the
    other. Anything that is not an object once decoded — a list, a bare string,
    malformed JSON — is not a set of named arguments, and reading it as one would
    mean guessing which key a model meant. A provider that produces one is
    reported as a malformed call rather than dispatched.
    """
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ToolCallError(f"{name} arguments are not valid JSON") from exc
    if not isinstance(arguments, dict):
        raise ToolCallError(f"{name} arguments are not a JSON object")
    return ToolCall(id=id, name=name, arguments=arguments)


def render_tool_result(call: ToolCall, content: str) -> ToolMessage:
    """Return the tool entry that answers one call, addressed by its id."""
    return ToolMessage(role="tool", text=content, tool_call_id=call.id, name=call.name)


def user_message(text: str) -> ToolMessage:
    """Return the user's turn."""
    return ToolMessage(role="user", text=text)


def assistant_message(text: str, calls: Sequence[ToolCall]) -> ToolMessage:
    """Return the model's turn, with whatever it wrote and everything it asked for."""
    return ToolMessage(role="assistant", text=text, tool_calls=tuple(calls))


class ToolUseUnsupportedError(RuntimeError):
    """The bound provider cannot run a tool-using call."""
