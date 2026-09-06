"""
Groq chat provider.

Clients are built per API key (cached) so every user's stored key can be
used without process-global state.
"""

from functools import lru_cache
from typing import Iterator

from groq import Groq

from services.llm.base import LLMProvider
from services.prompts import SYSTEM_INSTRUCTION


@lru_cache(maxsize=32)
def _client(api_key: str) -> Groq:
    """Do client."""
    return Groq(api_key=api_key)


class GroqProvider(LLMProvider):
    """Chat through the Groq SDK."""

    name = "groq"

    def _build_client(self):
        return _client(self.api_key)

    def _generate_response(self, query: str, context: str) -> str:
        """Do generate response."""
        chat_completion = self._sdk_client().chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_INSTRUCTION,
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuery: {query}",
                },
            ],
            model=self.model,
        )

        result = chat_completion.choices[0].message.content
        return result or self.FALLBACK_ANSWER

    def _stream_response(self, query: str, context: str) -> Iterator[str]:
        """Do stream response."""
        stream = self._sdk_client().chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_INSTRUCTION,
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuery: {query}",
                },
            ],
            model=self.model,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

    def verify(self) -> None:
        """Do verify."""
        self._sdk_client().chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
