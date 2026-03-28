from google.genai import types
from config import genai_client, EMBEDDING_MODEL, CHAT_MODEL

class AIService:
    @staticmethod
    def get_embeddings(texts):
        if isinstance(texts, str):
            texts = [texts]
        
        result = genai_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=768),
        )
        return [embedding.values for embedding in result.embeddings]

    @staticmethod
    def generate_response(query, context):
        system_instruction = (
            "You must only answer questions based on the provided context. "
            "If the context does not contain the answer, say 'I don't know based on the given context.' "
            "Do not use any outside knowledge."
        )

        try:
            result = genai_client.models.generate_content(
                model=CHAT_MODEL,
                config=types.GenerateContentConfig(system_instruction=system_instruction),
                contents=[
                    f"Context: {context}",
                    query,
                ],
            )
        except Exception:
            if context and context.strip():
                return (
                    "I couldn't use the language model right now, so here is relevant context from your document:\n\n"
                    f"{context[:1200]}"
                )
            return "I don't know based on the given context."

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
