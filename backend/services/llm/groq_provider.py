"""
Groq chat provider.

Clients are built per API key (cached) so the stored key can be used
without process-global state. Every call carries the model's output budget,
and the stream stops as soon as Groq reports a finish reason it knows, so a
truncated answer is never padded out past the reason that ended it.
"""

from functools import lru_cache
from typing import Iterator

from groq import Groq

from services.llm.base import LLMProvider, normalize_finish_reason
from services.prompts import SYSTEM_INSTRUCTION, build_user_prompt


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
