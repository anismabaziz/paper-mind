"""
Google chat provider.

Clients are built per API key (cached) so the stored key can be used
without process-global state. Every call carries the model's output budget,
and the stream stops as soon as Gemini reports a finish reason it knows, so
a truncated answer is never padded out past the reason that ended it.

Tool calls are translated to and from Gemini's own vocabulary here, so the
brief loop above never sees a ``FunctionCall``. Gemini has no separate tool
message role: a tool result is a ``function_response`` part inside a user
turn, which is how the protocol pairs a result with the call that asked for
it.
"""

import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Iterator

from google import genai
from google.genai import types

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
    """Evict all cached Google clients (called when keys rotate or tests reset)."""
    _client.cache_clear()


@lru_cache(maxsize=32)
def _client(api_key: str) -> genai.Client:
    """Do client."""
    return genai.Client(api_key=api_key)


class GoogleProvider(LLMProvider):
    """Chat through the Google GenAI SDK."""

    name = "google"

    def _build_client(self):
        if not self._use_cache:
            return genai.Client(api_key=self.api_key)
        return _client(self.api_key)

    def _config(self) -> "types.GenerateContentConfig":
        """Return the generation config one call runs under."""
        return types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            max_output_tokens=self.budget.max_output_tokens,
        )

    def _generate_response(
        self, query: str, context: str, prior_turns: str = ""
    ) -> str:
        """Do generate response."""
        result = self._sdk_client().models.generate_content(
            model=self.model,
            config=self._config(),
            contents=[build_user_prompt(context, query, prior_turns)],
        )

        text = getattr(result, "text", None)
        if text:
            return text

        candidates = getattr(result, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            collected_parts = []
            for part in parts:
                part_text = getattr(part, "text", None)
                if part_text:
                    collected_parts.append(part_text)
            if collected_parts:
                return "\n".join(collected_parts)

        return self.FALLBACK_ANSWER

    def _stream_response(
        self, query: str, context: str, prior_turns: str = ""
    ) -> Iterator[str]:
        """Do stream response."""
        for chunk in self._sdk_client().models.generate_content_stream(
            model=self.model,
            config=self._config(),
            contents=[build_user_prompt(context, query, prior_turns)],
        ):
            # Gemini can put the last fragment and the reason on one chunk, so
            # the text is taken before the reason is read.
            text = getattr(chunk, "text", None)
            if text:
                yield text
            for candidate in getattr(chunk, "candidates", None) or []:
                reason = normalize_finish_reason(
                    getattr(candidate, "finish_reason", None)
                )
                if reason is None:
                    continue
                self.last_finish_reason = reason
                if reason in self.budget.finish_reasons:
                    return

    @staticmethod
    def _declarations(tools: Sequence[ToolSpec]) -> list[types.FunctionDeclaration]:
        """Return the brief's tools as the function declarations Gemini takes."""
        return [
            types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                # The SDK types the schema as its own object, but accepts the
                # plain mapping the brief declared, so the brief validates
                # arguments against one schema and this sends that same one
                # rather than a translation of it.
                parameters=tool.parameters,  # type: ignore[arg-type]
            )
            for tool in tools
        ]

    @staticmethod
    def _contents(messages: Sequence[ToolMessage]) -> list[types.Content]:
        """
        Return the conversation as Gemini contents.

        Gemini carries a tool result as a ``function_response`` part on a user
        turn, so a tool entry becomes its own content rather than joining the
        model turn that asked for it. That is the API's own pairing, and it is
        why the ids are kept: a result addressed to the wrong call is a result
        the model cannot read.
        """
        contents: list[types.Content] = []
        for message in messages:
            if message.role == "user":
                contents.append(
                    types.Content(role="user", parts=[types.Part(text=message.text)])
                )
            elif message.role == "assistant":
                parts: list[types.Part] = []
                if message.text:
                    parts.append(types.Part(text=message.text))
                parts.extend(
                    types.Part(
                        function_call=types.FunctionCall(
                            id=call.id, name=call.name, args=dict(call.arguments)
                        )
                    )
                    for call in message.tool_calls
                )
                contents.append(types.Content(role="model", parts=parts))
            else:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                function_response=types.FunctionResponse(
                                    id=message.tool_call_id,
                                    name=message.name,
                                    response={"result": message.text},
                                )
                            )
                        ],
                    )
                )
        return contents

    @staticmethod
    def _usage(result: Any, field_name: str) -> int:
        """Return one token count the SDK reported, or zero where it reported none."""
        metadata = getattr(result, "usage_metadata", None)
        value = getattr(metadata, field_name, None)
        return int(value) if isinstance(value, (int, float)) else 0

    def _read(self, result: Any) -> ToolTurn:
        """
        Read one Gemini response as a provider-neutral tool turn.

        A part the SDK gave us with a name but unreadable arguments is dropped
        and logged rather than dispatched: guessing which key a model meant is
        how a search tool ends up querying a document it was not scoped to.
        """
        candidates = getattr(result, "candidates", None) or []
        texts: list[str] = []
        calls: list[ToolCall] = []
        finish_reason: str | None = None
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    texts.append(part_text)
                function_call = getattr(part, "function_call", None)
                if function_call is None:
                    continue
                try:
                    calls.append(
                        tool_call(
                            id=getattr(function_call, "id", None) or "",
                            name=str(getattr(function_call, "name", "")),
                            arguments=getattr(function_call, "args", None) or {},
                        )
                    )
                except ToolCallError as exc:
                    log.warning(
                        "%s tool call not dispatched (%s): %s",
                        self.name,
                        type(exc).__name__,
                        exc,
                    )
            reason = normalize_finish_reason(getattr(candidate, "finish_reason", None))
            if reason is not None:
                finish_reason = reason
        text = getattr(result, "text", None) or "\n".join(texts)
        return ToolTurn(
            text=text,
            tool_calls=tuple(calls),
            finish_reason=finish_reason,
            input_tokens=self._usage(result, "prompt_token_count"),
            output_tokens=self._usage(result, "candidates_token_count"),
        )

    def _complete_with_tools(
        self,
        messages: Sequence[ToolMessage],
        tools: Sequence[ToolSpec],
        system_instruction: str,
    ) -> ToolTurn:
        """Run one tool-using turn through the Gemini SDK."""
        result = self._sdk_client().models.generate_content(
            model=self.model,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction or SYSTEM_INSTRUCTION,
                max_output_tokens=self.budget.max_output_tokens,
                tools=[types.Tool(function_declarations=self._declarations(tools))],
            ),
            contents=self._contents(messages),
        )
        return self._read(result)

    def verify(self) -> None:
        """Verify the key with the same framing chat uses, under a timeout."""
        client = self._sdk_client()

        def _ping():
            return client.models.generate_content(
                model=self.model,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION, max_output_tokens=1
                ),
                contents=[build_user_prompt("", "ping")],
            )

        self._verify_with_timeout(_ping)
