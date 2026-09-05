"""
Groq chat provider.

Clients are built per API key (cached) so every user's stored key can be
used without process-global state.
"""

from functools import lru_cache

from groq import Groq

from services.prompts import SYSTEM_INSTRUCTION


@lru_cache(maxsize=32)
def _client(api_key: str) -> Groq:
    """Do client."""
    return Groq(api_key=api_key)


class GroqService:
    """GroqService."""

    @staticmethod
    def generate_response(query: str, context: str, api_key: str, model: str) -> str:
        """Do generate response."""
        chat_completion = _client(api_key).chat.completions.create(
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
            model=model,
        )

        result = chat_completion.choices[0].message.content
        return result or "I don't know based on the given context."

    @staticmethod
    def stream_response(query: str, context: str, api_key: str, model: str):
        """Do stream response."""
        stream = _client(api_key).chat.completions.create(
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
            model=model,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
