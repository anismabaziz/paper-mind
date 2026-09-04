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
    def _providers():
        # MODE picks the primary provider; the other one is the fallback.
        primary = config.MODE if config.MODE in ("groq", "google") else "google"
        fallback = "google" if primary == "groq" else "groq"
        return primary, fallback

    @staticmethod
    def _stream_with(provider: str, query: str, context: str):
        if provider == "groq":
            return GroqService.stream_response(query, context)
        return GoogleService.stream_response(query, context)

    @staticmethod
    def stream_response(query: str, context: str):
        """
            Yield answer fragments from the primary provider.

                    If the primary provider fails before producing any output, the
                    fallback provider answers instead. A failure that happens
                    mid-stream is re-raised so the caller can surface it.
        """
        primary, fallback = AIService._providers()
        emitted = False
        try:
            for token in AIService._stream_with(primary, query, context):
                emitted = True
                yield token
            return
        except Exception as e:
            print(f"AI Streaming Error ({primary}): {e}")
            if emitted:
                # Partial answer already streamed; replaying via the
                # fallback would duplicate or contradict it.
                raise

        # Primary never produced a token — let the fallback answer.
        try:
            yield from AIService._stream_with(fallback, query, context)
        except Exception as e:
            print(f"AI Streaming Error ({fallback}): {e}")
            raise

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
