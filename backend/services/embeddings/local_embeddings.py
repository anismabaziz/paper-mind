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

Tests never load the real model: a fake encoder is injected through the
constructor, and the import of ``sentence_transformers`` is lazy so
``pytest`` does not require the package or a network call. When the package
is not installed, a clear error is raised only when embeddings are actually
generated.
"""

import threading

from services.concurrency import map_batches_concurrently

# Local BGE-M3 has no provider-side cap; 100 keeps CPU peak memory sane.
EMBED_BATCH_SIZE = 100


class LocalEmbeddingService:
    """
    CPU embedding via BAAI/bge-m3.

    The model name arrives through the constructor; the model itself loads
    lazily on first use and is cached on the instance. The public entry is
    ``embed_texts`` (batched, concurrent); tests inject a fake encoder
    through the constructor's ``model`` argument to avoid loading weights.
    """

    def __init__(self, model_name: str, device: str = "cpu", model=None):
        """
        Bind the model name; the model itself loads lazily.

        ``model`` injects a pre-loaded (or fake) encoder; when omitted, the
        real model loads lazily on first use.
        """
        self._model_name = model_name
        self._device = device
        self._model = model
        self._lock = threading.Lock()

    def _get_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for embedding. "
                    "Install it with `uv sync` or `pip install sentence-transformers`."
                ) from exc

            # ``device="cpu"`` keeps the free path CPU-only; no CUDA needed.
            # ``trust_remote_code=True`` is required for BGE-M3's custom code.
            self._model = SentenceTransformer(
                self._model_name, trust_remote_code=True, device=self._device
            )
            return self._model

    def embed_texts(self, texts):
        """
        Embed any number of texts, batched and run concurrently.

        Accepts a single string (query path) or a list (document path) and
        returns one vector per input text, in input order.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return []

        batches = [
            texts[i : i + EMBED_BATCH_SIZE]
            for i in range(0, len(texts), EMBED_BATCH_SIZE)
        ]
        results = map_batches_concurrently(
            batches,
            self._embed_batch,
            label=f"embed_texts: {len(texts)} texts",
        )
        all_values: list = []
        for result in results:
            all_values.extend(result)
        return all_values

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed one batch of texts with BGE-M3, normalized for cosine distance.

        Returns a list of 1024-d vectors (Matryoshka-truncated when the model
        exposes ``truncate_dim``).
        """
        model = self._get_model()
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
            if (
                hasattr(embeddings, "shape")
                and len(embeddings.shape) == 2
                and embeddings.shape[1] > 1024
            ):
                embeddings = embeddings[:, :1024]

        # Convert numpy -> python lists for the vector index contract.
        # ``tolist`` keeps dtype float; Qdrant accepts python floats.
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()
        return [list(row) for row in embeddings]
