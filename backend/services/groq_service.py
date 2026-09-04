import os

import config
from services.prompts import SYSTEM_INSTRUCTION

GROQ_MODEL = config.GROQ_MODEL

class GroqService:
    @staticmethod
    def generate_response(query: str, context: str) -> str:
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("Groq client is not initialized. Please ensure GROQ_API_KEY is configured in your .env file.")

        chat_completion = config.groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_INSTRUCTION,
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuery: {query}",
                }
            ],
            model=GROQ_MODEL,
        )

        result = chat_completion.choices[0].message.content
        return result or "I don't know based on the given context."

    @staticmethod
    def stream_response(query: str, context: str):
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("Groq client is not initialized. Please ensure GROQ_API_KEY is configured in your .env file.")

        stream = config.groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_INSTRUCTION,
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuery: {query}",
                }
            ],
            model=GROQ_MODEL,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta
