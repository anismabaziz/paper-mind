from google.genai import types
import config
from services.prompts import SYSTEM_INSTRUCTION

CHAT_MODEL = config.CHAT_MODEL

class GoogleService:
    @staticmethod
    def generate_response(query: str, context: str) -> str:
        result = config.genai_client.models.generate_content(
            model=CHAT_MODEL,
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
    def stream_response(query: str, context: str):
        for chunk in config.genai_client.models.generate_content_stream(
            model=CHAT_MODEL,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
            contents=[
                f"Context: {context}",
                query,
            ],
        ):
            text = getattr(chunk, "text", None)
            if text:
                yield text
