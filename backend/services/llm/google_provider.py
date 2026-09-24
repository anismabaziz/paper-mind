"""
Google chat provider.

Clients are built per API key (cached) so the stored key can be used
without process-global state.
"""

from functools import lru_cache
from typing import Iterator

from google import genai
from google.genai import types

from services.llm.base import LLMProvider
from services.prompts import SYSTEM_INSTRUCTION, build_user_prompt


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

    def _generate_response(self, query: str, context: str) -> str:
        """Do generate response."""
        result = self._sdk_client().models.generate_content(
            model=self.model,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
            contents=[build_user_prompt(context, query)],
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

    def _stream_response(self, query: str, context: str) -> Iterator[str]:
        """Do stream response."""
        for chunk in self._sdk_client().models.generate_content_stream(
            model=self.model,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
            contents=[build_user_prompt(context, query)],
        ):
            text = getattr(chunk, "text", None)
            if text:
                yield text

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
