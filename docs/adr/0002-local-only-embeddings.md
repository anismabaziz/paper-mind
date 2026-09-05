# Local-only embeddings via BGE-M3

Embeddings are generated exclusively by the local `BAAI/bge-m3` model
(sentence-transformers, CPU, 8192-token context, 1024-dim Matryoshka,
normalized for cosine) — there is no online embedding path. The earlier Gemini
768-dim embedder was deleted: it contradicted the goal of running without
online services, and its output dimensionality clashed with the 1024-dim
Qdrant collection geometry, a latent mismatch waiting for an ingest. The
embedding dispatch seam is gone too — generation delegates directly to the
local embedding service, so 1024 is the only geometry the code can produce.
Trade-off: model weights (~2GB) download on first run and ingestion runs on
CPU, but ingestion works offline and never consumes embedding API quota.
