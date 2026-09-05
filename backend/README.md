# PaperMind Backend

Flask API for RAG chat over uploaded PDFs. Defaults to free local components
(Qdrant + BGE-M3) so a reviewer can run without creating any account.

## Free local vs keys

Same table as the top-level README (kept here so env docs stay local):

- `VECTOR_BACKEND=qdrant` (default, `http://localhost:6333`, compose `qdrant` service) vs `pinecone` (+ `PINECONE_API_KEY`)
- `EMBED_BACKEND=local` (default, `BAAI/bge-m3` 1024d, no key) vs `gemini` (`gemini-embedding-001` + `GOOGLE_API_KEY`)
- `RERANK=false` (default) / `true` — local cross-encoder over 50→5 (`cross-encoder/ms-marco-MiniLM-L-6-v2` 22M fast default, or `BAAI/bge-reranker-v2-m3`)
- `CHUNK_SIZE_TOKENS=512` / `CHUNK_OVERLAP_TOKENS=50` via `tiktoken cl100k_base`, per-page, with `page_no` + `content_hash`
- Parser seam: `pymupdf` fast path default; `USE_DOCLING=auto` (default) routes only image-only / borderless-table / 2-col PDFs to Docling (`.[docling]` extra, `granite-docling-258M`); `true` forces all, `false` never.
- All four knobs are documented in `.env.example` and wired in `compose.yaml`.

## Setup (one command, from a clean clone)

Free local (no keys):

```bash
# from the repo root; DEMO_MODE skips login for a first look
DEMO_MODE=true docker compose up --build
```

Keys path (opt-in):

```bash
export VECTOR_BACKEND=pinecone PINECONE_API_KEY=...
export EMBED_BACKEND=gemini GOOGLE_API_KEY=...   # or keep local
# MODE=google needs GOOGLE_API_KEY, MODE=groq needs GROQ_API_KEY
docker compose up --build
```

Either path boots Postgres (+ Qdrant when `VECTOR_BACKEND=qdrant`), runs the
Alembic migrations, and serves the API on `http://127.0.0.1:3000`
(`GET /health` to check).

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

## Auth

With `DEMO_MODE=false`, user-facing endpoints require a JWT: register with
`POST /auth/register` (`{"email", "password"}`), log in with
`POST /auth/login`, and send the returned token as
`Authorization: Bearer <token>`. Passwords are bcrypt-hashed. Setting
`DEMO_MODE=true` disables the checks so the app is fully usable without
logging in. `JWT_SECRET` signs tokens; unset, a per-process random secret is
used (fine for demos, set it for shared deployments).

## Tests

```bash
uv run pytest
```

Tests never talk to real Pinecone, LLM, Postgres, or the real upload
directory — `tests/conftest.py` provides dummy environment values, and the
flow tests in `tests/test_flows.py` run against fakes and in-memory sqlite.

## Evaluation

`evaluation/` measures retrieval and answer quality against a committed
ground-truth fixture (`evaluation/fixture.json`): ten questions over two
sample documents in `evaluation/sample_docs/` — one authored in-repo
(CC0), one published paper (CC BY 4.0). The evaluator reports
`hit@5`/`recall@5` (k=5, `FETCH_K=10` internally) + per-question breakdown and
ingest `sec/PDF` (parse/embed/upsert wall time via `evaluation/evaluator.py`;
`POST /process-file` also logs `parse/embed/upsert/total` per file).
`uv run pytest` exercises the scoring on deterministic fakes and stays
headless (no Qdrant/Pinecone/LLM, heavy models mocked).

Live run — opt-in because it writes into the real vector index (and the
judge costs LLM calls):

```bash
cd backend
# Free local path: Qdrant on http://localhost:6333, no Pinecone/Google keys needed
# (requires: docker compose up qdrant, or QDRANT_URL=http://localhost:6333,
#  VECTOR_BACKEND=qdrant + EMBED_BACKEND=local — both are the defaults)
uv run python -m evaluation.cli --live --no-judge          # retrieval only, no LLM key
uv run python -m evaluation.cli --live                     # + LLM-as-judge faithfulness (needs MODE key)
uv run python -m evaluation.cli --live --json              # machine-readable
uv run python -m evaluation.cli --live --rerank            # force RERANK=true (local cross-encoder 50→5)
uv run python -m evaluation.cli --live --compare-rerank    # with vs without reranker + latency delta

# With keys (Pinecone/Gemini) — same CLI, just flip env:
VECTOR_BACKEND=pinecone PINECONE_API_KEY=... EMBED_BACKEND=gemini GOOGLE_API_KEY=... uv run python -m evaluation.cli --live
```

A live run indexes the sample docs under an `eval-` prefix in the vector
index and deletes them afterwards. When `VECTOR_BACKEND=qdrant` (default)
the index lives at `http://localhost:6333` (compose exposes 6333→6333 and
6334→6334, app uses `QDRANT_URL=http://qdrant:6333` inside compose);
no Pinecone or Google key is required for retrieval-only (`--no-judge`)
when `EMBED_BACKEND=local`.

