"""
Google chat provider.

Clients are built per API key (cached) so every user's stored key can be
used without process-global state.
"""

from functools import lru_cache

from google import genai
from google.genai import types

from services.prompts import SYSTEM_INSTRUCTION


@lru_cache(maxsize=32)
def _client(api_key: str) -> genai.Client:
    """Do client."""
    return genai.Client(api_key=api_key)


class GoogleService:
    """GoogleService."""

    @staticmethod
    def generate_response(query: str, context: str, api_key: str, model: str) -> str:
        """Do generate response."""
        result = _client(api_key).models.generate_content(
            model=model,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
            contents=[
                f"Context: {context}",
                query,
            ],
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

        return "I don't know based on the given context."

    @staticmethod
    def stream_response(query: str, context: str, api_key: str, model: str):
        """Do stream response."""
        for chunk in _client(api_key).models.generate_content_stream(
            model=model,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
            contents=[
                f"Context: {context}",
                query,
            ],
        ):
            text = getattr(chunk, "text", None)
            if text:
                yield text
