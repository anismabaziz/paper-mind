import re
import time

import config
from google.genai import types
from services.concurrency import map_batches_concurrently
from services.google_service import GoogleService
from services.groq_service import GroqService
from services.local_embeddings import LocalEmbeddingService

class AIService:
    # Gemini BatchEmbedContents is capped at 100 contents per request
    EMBED_BATCH_SIZE = 100

    @staticmethod
    def _embed_batch_with_retry(batch, max_retries=3):
        last_exc = None
        for attempt in range(max_retries):
            try:
                result = config.genai_client.models.embed_content(
                    model=config.EMBEDDING_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(output_dimensionality=768),
                )
                return result
            except Exception as exc:
                last_exc = exc
                msg = str(exc)
                # 429 quota – respect RetryInfo if present, else exponential backoff
                is_rate_limit = "429" in msg or "RESOURCE_EXHAUSTED" in msg or "Quota exceeded" in msg
                if is_rate_limit and attempt < max_retries - 1:
                    m = re.search(r"retry in ([\d.]+)s", msg)
                    delay = float(m.group(1)) + 1 if m else (2 ** attempt) * 2
                    # cap so a single batch never blocks longer than ~60s
                    delay = min(delay, 60)
                    print(f"Embedding rate-limited, retry {attempt + 1}/{max_retries} after {delay:.1f}s: {exc}")
                    time.sleep(delay)
                    continue
                raise
        raise last_exc  # pragma: no cover

    @staticmethod
    def get_embeddings(texts):
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return []

        embed_backend = config._embed_backend()
        batches = [texts[i : i + AIService.EMBED_BATCH_SIZE] for i in range(0, len(texts), AIService.EMBED_BATCH_SIZE)]

        # Single dispatch map eliminates the repeated if/else cascade and the
        # middle-man _local_embed_batch (callers invoke the real target directly).
        dispatch = {
            config.EmbedBackend.LOCAL.value: (LocalEmbeddingService._embed_batch, "local", lambda r: r),
            config.EmbedBackend.GEMINI.value: (
                AIService._embed_batch_with_retry,
                "gemini",
                lambda r: (embedding.values for embedding in r.embeddings),
            ),
        }
        func, label_suffix, unpack = dispatch.get(embed_backend, dispatch[config.EmbedBackend.LOCAL.value])
        results = map_batches_concurrently(
            batches,
            func,
            label=f"AIService.get_embeddings[{label_suffix}]: {len(texts)} texts",
        )
        all_values: list = []
        for result in results:
            all_values.extend(unpack(result))
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
