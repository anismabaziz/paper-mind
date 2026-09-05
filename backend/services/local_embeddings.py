"""
Local embedding service via BAAI/bge-m3 (sentence-transformers).

Default free path — no API key, runs on CPU. Heavy weights (~2GB) are
downloaded on first use and cached to the Hugging Face cache directory
(``HF_HOME`` / ``~/.cache/huggingface``). When running via Docker, mount
that cache to a persistent volume so the download only happens once.

The model supports 8192 context and Matryoshka truncation to 1024d, which
matches the Qdrant collection (``qdrant_store._QDRANT_DENSE_SIZE``). Keeping
dimensions at 1024 keeps storage and latency low (~15ms vs ~42ms at 3072)
while preserving quality.

Tests never load the real model: ``AIService.get_embeddings`` is monkeypatched
or ``LocalEmbeddingService._embed_batch`` is stubbed, and the import of
``sentence_transformers`` is lazy so ``pytest`` does not require the package
or a network call. When the package is not installed, a clear error is raised
only when the local backend is actually selected.
"""

import os
import threading

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for EMBED_BACKEND=local. "
                "Install it with `uv sync` or `pip install sentence-transformers`, "
                "or switch to EMBED_BACKEND=gemini."
            ) from exc

        model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-m3")
        # ``device="cpu"`` keeps the free path CPU-only; no CUDA needed.
        # ``trust_remote_code=True`` is required for BGE-M3's custom code.
        _model = SentenceTransformer(model_name, trust_remote_code=True, device="cpu")
        return _model


class LocalEmbeddingService:
    """
    CPU embedding via BAAI/bge-m3.

    The public entry is ``_embed_batch`` (one batch) so ``AIService`` can
    reuse its existing concurrent batcher and retry shape. Tests stub this
    method to avoid loading weights.
    """

    @staticmethod
    def _embed_batch(texts: list[str]) -> list[list[float]]:
        """
        Embed one batch of texts with BGE-M3, normalized for cosine distance.

        Returns a list of 1024-d vectors (Matryoshka-truncated when the model
        exposes ``truncate_dim``).
        """
        model = _get_model()
        # Prefer Matryoshka truncation to 1024 when supported.
        try:
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=len(texts),
                show_progress_bar=False,
                convert_to_numpy=True,
                truncate_dim=1024,
            )
        except TypeError:
            # Older sentence-transformers without truncate_dim: encode full then slice.
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=len(texts),
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            # embeddings is numpy array of shape (n, dim)
            if hasattr(embeddings, "shape") and len(embeddings.shape) == 2 and embeddings.shape[1] > 1024:
                embeddings = embeddings[:, :1024]

        # Convert numpy -> python lists for the vector index contract.
        # ``tolist`` keeps dtype float; Qdrant/pinecone accept python floats.
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()
        return [list(row) for row in embeddings]

    @staticmethod
    def _reset_for_tests():
        """Clear the cached model so tests can re-stub without leaking state."""
        global _model
        with _model_lock:
            _model = None
