"""
Groq chat provider.

Clients are built per API key (cached) so the stored key can be used
without process-global state. Every call carries the model's output budget,
and the stream stops as soon as Groq reports a finish reason it knows, so
a truncated answer is never padded out past the reason that ended it.

Tool calls are translated to and from Groq's own vocabulary here, so the
brief loop above never sees a ``ChatCompletionMessageToolCall``. Groq's
protocol is the OpenAI-shaped one: the model turn carries the calls, and
each result is its own ``tool`` message addressed by the call id.
"""

import json
import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Iterator

from groq import Groq

from services.llm.base import LLMProvider, normalize_finish_reason
from services.llm.tools import (
    ToolCall,
    ToolCallError,
    ToolMessage,
    ToolSpec,
    ToolTurn,
    tool_call,
)
from services.prompts import SYSTEM_INSTRUCTION, build_user_prompt

log = logging.getLogger(__name__)


def clear_cache() -> None:
    """Evict all cached Groq clients (called when keys rotate or tests reset)."""
    _client.cache_clear()


@lru_cache(maxsize=32)
def _client(api_key: str) -> Groq:
    """Do client."""
    return Groq(api_key=api_key)


class GroqProvider(LLMProvider):
    """Chat through the Groq SDK."""

    name = "groq"

    def _build_client(self):
        if not self._use_cache:
            return Groq(api_key=self.api_key)
        return _client(self.api_key)

    def _messages(self, query: str, context: str, prior_turns: str) -> list[dict]:
        """Return the system and user messages one call sends."""
        return [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": build_user_prompt(context, query, prior_turns)},
        ]

    def _generate_response(
        self, query: str, context: str, prior_turns: str = ""
    ) -> str:
        """Do generate response."""
        chat_completion = self._sdk_client().chat.completions.create(
            messages=self._messages(query, context, prior_turns),
            model=self.model,
            max_tokens=self.budget.max_output_tokens,
        )

        result = chat_completion.choices[0].message.content
        return result or self.FALLBACK_ANSWER

    def _stream_response(
        self, query: str, context: str, prior_turns: str = ""
    ) -> Iterator[str]:
        """Do stream response."""
        stream = self._sdk_client().chat.completions.create(
            messages=self._messages(query, context, prior_turns),
            model=self.model,
            max_tokens=self.budget.max_output_tokens,
            stream=True,
        )

        for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if choice is None:
                continue
            # Groq can put the last fragment and the reason on one chunk, so
            # the text is taken before the reason is read.
            if choice.delta.content:
                yield choice.delta.content
            reason = normalize_finish_reason(choice.finish_reason)
            if reason is not None:
                self.last_finish_reason = reason
                if reason in self.budget.finish_reasons:
                    return

    @staticmethod
    def _declarations(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
        """Return the brief's tools as the function declarations Groq takes."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    @staticmethod
    def _tool_messages(
        messages: Sequence[ToolMessage], system_instruction: str
    ) -> list[dict[str, Any]]:
        """
        Return the conversation as Groq messages.

        Groq's protocol gives a tool result its own ``tool`` message addressed by
        the call id, so the ids are carried straight through: a result paired to
        the wrong call is a result the model reads as belonging to something else.
        Groq has no separate system-instruction field, so the instruction leads
        the conversation as a message, which is how its own chat path sends it.
        """
        sent: list[dict[str, Any]] = []
        if system_instruction:
            sent.append({"role": "system", "content": system_instruction})
        for message in messages:
            if message.role == "user":
                sent.append({"role": "user", "content": message.text})
            elif message.role == "assistant":
                sent.append(
                    {
                        "role": "assistant",
                        "content": message.text or None,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": json.dumps(call.arguments),
                                },
                            }
                            for call in message.tool_calls
                        ],
                    }
                )
            else:
                sent.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": message.text,
                    }
                )
        return sent

    def _read(self, completion: Any) -> ToolTurn:
        """
        Read one Groq completion as a provider-neutral tool turn.

        A call whose arguments are not a JSON object is dropped and logged rather
        than dispatched: a search tool handed an argument shape it did not expect
        is a search over a document it was not scoped to, or over all of them.
        """
        choices = getattr(completion, "choices", None) or []
        if not choices:
            return ToolTurn(finish_reason=None)
        choice = choices[0]
        message = getattr(choice, "message", None)
        calls: list[ToolCall] = []
        for declared in getattr(message, "tool_calls", None) or []:
            function = getattr(declared, "function", None)
            try:
                calls.append(
                    tool_call(
                        id=str(getattr(declared, "id", "") or ""),
                        name=str(getattr(function, "name", "")),
                        arguments=getattr(function, "arguments", None) or {},
                    )
                )
            except ToolCallError as exc:
                log.warning(
                    "%s tool call not dispatched (%s): %s",
                    self.name,
                    type(exc).__name__,
                    exc,
                )
        usage = getattr(completion, "usage", None)
        return ToolTurn(
            text=getattr(message, "content", None) or "",
            tool_calls=tuple(calls),
            finish_reason=normalize_finish_reason(
                getattr(choice, "finish_reason", None)
            ),
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )

    def _complete_with_tools(
        self,
        messages: Sequence[ToolMessage],
        tools: Sequence[ToolSpec],
        system_instruction: str,
    ) -> ToolTurn:
        """Run one tool-using turn through the Groq SDK."""
        completion = self._sdk_client().chat.completions.create(
            messages=self._tool_messages(messages, system_instruction),
            model=self.model,
            max_tokens=self.budget.max_output_tokens,
            tools=self._declarations(tools),
            tool_choice="auto",
        )
        return self._read(completion)

    def verify(self) -> None:
        """Verify the key with the same framing chat uses, under a timeout."""
        client = self._sdk_client()

        def _ping():
            return client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": build_user_prompt("", "ping")},
                ],
                max_tokens=1,
            )

        self._verify_with_timeout(_ping)
