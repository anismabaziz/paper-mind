"""Chat orchestration over the per-user provider settings."""

from services.concurrency import map_batches_concurrently
from services.google_service import GoogleService
from services.groq_service import GroqService
from services.local_embeddings import LocalEmbeddingService


class AIService:
    """AIService."""

    # Local BGE-M3 has no provider-side cap; 100 keeps CPU peak memory sane.
    EMBED_BATCH_SIZE = 100

    @staticmethod
    def get_embeddings(texts):
        """Do get embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return []

        batches = [
            texts[i : i + AIService.EMBED_BATCH_SIZE]
            for i in range(0, len(texts), AIService.EMBED_BATCH_SIZE)
        ]
        results = map_batches_concurrently(
            batches,
            LocalEmbeddingService._embed_batch,
            label=f"AIService.get_embeddings: {len(texts)} texts",
        )
        all_values: list = []
        for result in results:
            all_values.extend(result)
        return all_values

    @staticmethod
    def _stream_with(provider: str, query: str, context: str, api_key: str, model: str):
        # Direct dispatch (not a pre-bound dict) so monkeypatched
        # GroqService/GoogleService in tests is respected.
        if provider == "groq":
            return GroqService.stream_response(query, context, api_key, model)
        return GoogleService.stream_response(query, context, api_key, model)

    @staticmethod
    def stream_response(
        query: str, context: str, provider: str, model: str, api_key: str
    ):
        """
        Yield answer fragments from the user's chosen provider.

        Failures propagate: the caller surfaces them as an SSE error event.
        There is no cross-provider fallback — the user picked this provider.
        """
        yield from AIService._stream_with(provider, query, context, api_key, model)

    @staticmethod
    def generate_response(
        query: str, context: str, provider: str, model: str, api_key: str
    ) -> str:
        """Do generate response."""
        try:
            if provider == "groq":
                return GroqService.generate_response(query, context, api_key, model)
            return GoogleService.generate_response(query, context, api_key, model)
        except Exception as e:
            print(f"AI Generation Error ({provider}): {e}")
            if context and context.strip():
                return (
                    "I couldn't use the language model right now, so here is relevant context from your document:\n\n"
                    f"{context[:1200]}"
                )
            return "I don't know based on the given context."
