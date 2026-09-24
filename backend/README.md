# PaperMind Backend

Flask API for RAG chat over uploaded PDFs. Single-instance, no auth — one
`app_settings` row holds the BYO provider key.

## Free local vs keys

Same table as the top-level README (kept here so env docs stay local):

- Qdrant on `http://localhost:6333` (compose `qdrant` service), no API key
- Embeddings: `BAAI/bge-m3` 1024d via `sentence-transformers`, CPU, no API key
- Retrieval: named dense vectors plus hashed term-frequency sparse vectors with Qdrant IDF, fused by one query using `RRF(k=60)` and 50 candidates
- `RERANK=false` (default) / `true` — local cross-encoder over 50→5 (`cross-encoder/ms-marco-MiniLM-L-6-v2` 22M fast default, or `BAAI/bge-reranker-v2-m3`)
- `CHUNK_SIZE_TOKENS=512` / `CHUNK_OVERLAP_TOKENS=50` via `tiktoken cl100k_base`, per-page, with `page_no` + `content_hash`
- Parser: `pymupdf` fast path default; `USE_DOCLING=auto` (default) routes only image-only / borderless-table / 2-col PDFs to Docling (`.[docling]` extra, `granite-docling-258M`); `true` forces all, `false` never.
- All knobs are documented in `.env.example`.

## Setup (one command, from a clean clone)

```bash
# from the repo root — start infra
docker compose -f backend/compose.yaml up -d
# then run backend locally
cd backend
uv sync
uv run alembic upgrade head
uv run python app.py   # API on http://127.0.0.1:3000 (GET /health)
```

The only required env vars are `DATABASE_URL` and `QDRANT_URL` (defaults to
`http://localhost:6333`). Optional: `APP_SECRET` — the Fernet root that
encrypts the stored provider key. Set it in any persistent deployment;
changing it invalidates previously stored keys. With no `APP_SECRET`, a
warning is printed and stored keys cannot be encrypted or decrypted. After boot, open
Settings in the app and paste your provider key.

Keys path: provider, model, and API key are stored as a single global
`app_settings` row (encrypted with `APP_SECRET`), configured through the
app's Settings dialog — not in the environment.

## Chat provider settings

There are no provider env vars. Chat provider, model, and API key are global
app settings stored encrypted in Postgres and configured through the app's
Settings dialog:

- `GET /settings` — current settings (masked key) plus the supported
  provider → models map
- `PUT /settings` — validate provider/model and encrypt the key
- `POST /settings/verify` — one-token completion against the chosen
  provider/model with the stored key

All three are open (no auth) and operate on the single `app_settings` row.
Chat runs on those global settings — a workspace with no saved settings gets
a clear "configure a provider in Settings" error, and the backend boots fine
with no keys at all. Retrieval-only evaluation (`evaluation.cli --live
--no-judge`) needs no chat key either.

## Setup (manual, without Docker)

```bash
cd backend
uv sync
cp .env.example .env   # then fill in the values
```

You need a Postgres database; point `DATABASE_URL` at it and create the
schema with the migrations:

```bash
uv run alembic upgrade head
```

On boot the app validates its environment: any missing required variable
is named on stderr with a pointer to `.env.example`, instead of a library
traceback.

## Run

```bash
uv run python app.py
```

## Tests

The default suite is fast and uses fakes with in-memory SQLite:

```bash
uv run pytest
```

The full-stack suite serves the Flask app over HTTP, starts pinned disposable Postgres and Qdrant containers, and applies every migration to a fresh database:

```bash
./run-full-stack-tests.sh
```

Its embedding, reranking, and chat providers are deterministic and local. The test uploads and parses the sample PDF, indexes it in Qdrant, streams an answer, inspects Postgres and Qdrant state, injects service failures, and removes all test data and containers on exit. No model API key is required.

## Evaluation

`evaluation/` measures retrieval and answer quality against a committed
ground-truth fixture (`evaluation/fixture.json`): ten questions over two
sample documents in `evaluation/sample_docs/` — one authored in-repo
(CC0), one published paper (CC BY 4.0). The evaluator reports
`hit@5`/`recall@5` (k=5, 50 candidates fetched internally) + per-question breakdown and
ingest `sec/PDF` (parse/embed/upsert wall time via `evaluation/evaluator.py`;
`POST /process-file` also logs `parse/embed/upsert/total` per file).
`uv run pytest` exercises the scoring on deterministic fakes and stays
headless (no Qdrant/LLM, heavy models mocked).

Live run — opt-in because it writes into the real vector index (and the
judge costs LLM calls):

```bash
cd backend
# Free local path: Qdrant on http://localhost:6333, no API key needed
# (requires: docker compose -f compose.yaml up -d qdrant, or QDRANT_URL=http://localhost:6333)
uv run python -m evaluation.cli --live --no-judge          # retrieval only, no LLM key
uv run python -m evaluation.cli --live                     # + LLM-as-judge faithfulness (needs a chat key)
uv run python -m evaluation.cli --live --json              # machine-readable
uv run python -m evaluation.cli --live --rerank            # force RERANK=true (local cross-encoder 50→5)
uv run python -m evaluation.cli --live --compare-rerank    # with vs without reranker + latency delta
```

A live run indexes the sample docs under an `eval-` prefix in the vector
index and deletes them afterwards. The index lives at `http://localhost:6333`
(compose exposes 6333→6333 and
6334→6334);
no chat key is required for retrieval-only (`--no-judge`).
