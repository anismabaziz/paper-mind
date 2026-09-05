"""Module docstring."""

import config
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
    def _providers():
        # MODE picks the primary provider; the other one is the fallback.
        # Read config.MODE directly so tests that monkeypatch it keep working;
        # _chat_provider is used for validation, not for streaming selection.
        primary = config.MODE if config.MODE in ("groq", "google") else "google"
        fallback = "google" if primary == "groq" else "groq"
        return primary, fallback

    @staticmethod
    def _stream_with(provider: str, query: str, context: str):
        # Direct dispatch (not a pre-bound dict) so monkeypatched
        # GroqService/GoogleService in tests is respected.
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
        """Do generate response."""
        try:
            provider = config.MODE if config.MODE in ("groq", "google") else "google"
            if provider == "groq":
                return GroqService.generate_response(query, context)
            return GoogleService.generate_response(query, context)
        except Exception as e:
            print(f"AI Generation Error ({config.MODE}): {e}")
            if context and context.strip():
                return (
                    "I couldn't use the language model right now, so here is relevant context from your document:\n\n"
                    f"{context[:1200]}"
                )
            return "I don't know based on the given context."
