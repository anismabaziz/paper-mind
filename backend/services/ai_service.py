import config
from google.genai import types
from services.google_service import GoogleService
from services.groq_service import GroqService

class AIService:
    @staticmethod
    def get_embeddings(texts):
        if isinstance(texts, str):
            texts = [texts]
        
        result = config.genai_client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=768),
        )
        return [embedding.values for embedding in result.embeddings]

    @staticmethod
    def generate_response(query: str, context: str) -> str:
        try:
            if config.MODE == "groq":
                return GroqService.generate_response(query, context)
            else:
                return GoogleService.generate_response(query, context)
        except Exception as e:
            print(f"AI Generation Error ({config.MODE}): {e}")
            if context and context.strip():
                return (
                    "I couldn't use the language model right now, so here is relevant context from your document:\n\n"
                    f"{context[:1200]}"
                )
            return "I don't know based on the given context."
