# PaperMind

Chat with your research papers. Upload a PDF, wait for it to index, then ask
questions and get answers streamed in from an LLM, each grounded in the
retrieved chunks of your document with the sources shown inline.

This is a portfolio project built to run locally. It is PDF-only by design:
the parser currently registers exactly one parser, for `.pdf`. There is
no hosted deployment and no multi-tenant story. What it does, it does on your
machine. The app runs as a single-instance workspace — no accounts, no login —
and stores one global set of provider settings.

## Features

- PDF upload, parsing, chunking, and indexing into a vector store
- Streaming chat over the indexed document (SSE), with retrieved sources
  attached to each answer
- Single-instance BYO provider: bring your own provider (Google or Groq),
  model, and API key via the Settings dialog — stored encrypted, never
  returned in plaintext
- Human titles derived from the PDF itself (metadata Title → original filename
  → first heading) while the stable uuid filename stays on disk and in Qdrant
- Retrieval evaluator with a committed ground-truth fixture
- Postgres for persistence, local filesystem for uploaded files

## Quick start

```bash
# Start infra (Postgres + Qdrant) — backend is not a compose service
docker compose -f backend/compose.yaml up -d
# Run backend locally
cd backend
uv sync
uv run alembic upgrade head
uv run python app.py   # API on http://127.0.0.1:3000 (GET /health)
```

The frontend is a Vite app and runs separately:

```bash
cd frontend
npm install
npm run dev
```

Then open the printed localhost URL. Environment variables for the frontend
are documented in [frontend/.env.example](frontend/.env.example), and the
backend's in [backend/.env.example](backend/.env.example).

Required env vars: `DATABASE_URL` (e.g. `postgresql+psycopg://papermind:papermind@localhost:5432/papermind`)
and `QDRANT_URL` (defaults to `http://localhost:6333`). Optional: `APP_SECRET`
— the Fernet root that encrypts the stored provider key. Set it in any
persistent deployment; changing it invalidates previously stored keys. With no
`APP_SECRET`, a warning is printed and stored keys cannot be encrypted or
decrypted. After boot, open Settings in the app and paste your provider key — no login step.

## Configuring a chat provider

There are no provider keys in the environment. Open the Settings dialog in the
app, pick a provider (Google or Groq) and a model from the curated list, paste
your own API key, and hit "Test connection" to verify before saving. Keys are
stored encrypted in the single `app_settings` row and applied to new chats
immediately. The app boots with no provider key present; asking a question
before saving settings returns a clear error pointing at Settings rather than
a crash or an env default.

Everything below runs free and local — no API keys anywhere in the pipeline
except the chat LLM, which is configured once in Settings:

| Concern | Detail |
|---|---|
| Vector store | Qdrant on `http://localhost:6333` (compose `qdrant` service, volume `qdrant_storage`), collection `pdf-index` |
| Embeddings | `BAAI/bge-m3` via `sentence-transformers`, CPU, 8192 ctx, 1024d Matryoshka, cached locally via `HF_HOME` (`~/.cache/huggingface`) — no API key |
| Retrieval | Hybrid 1,024-d dense + hashed term-frequency sparse with Qdrant IDF, fused by one Qdrant query using `RRF(k=60)`, 50 candidates → 5, gated reranker `RERANK=true` (22M MiniLM ~10ms/50 or `bge-reranker-v2-m3` ~80ms/50) |
| Chunking | `CHUNK_SIZE_TOKENS=512` / `CHUNK_OVERLAP_TOKENS=50` (~10%) via `tiktoken cl100k_base`, per-page, `page_no` + `content_hash` metadata |
| Parser | `pymupdf` fast path default; `USE_DOCLING=auto` routes only image-only / borderless-table / 2-col PDFs to Docling (opt-in `.[docling]`), `USE_DOCLING=true` forces all |
| Chat LLM | Single-instance Settings: provider (Google or Groq), curated model, your own API key — encrypted at rest via `APP_SECRET` |
| Document title | Derived from PDF metadata Title → original filename (without extension) → first heading; stored alongside the uuid `filename` |
| Evaluator live | `uv run python -m evaluation.cli --live --no-judge` works with just local Qdrant (no chat key); `--live` with the LLM-as-judge needs a key |

All free-path knobs live in `backend/.env.example`:
`RERANK`/`RERANK_MODEL`, `CHUNK_SIZE_TOKENS`/`CHUNK_OVERLAP_TOKENS`,
`USE_DOCLING`, `LOCAL_EMBEDDING_MODEL`.

The infra compose file is `backend/compose.yaml` (Postgres + Qdrant only).
Manual backend run (uv, local Postgres, Alembic) is in [backend/README.md](backend/README.md).

## Screenshots

![Main chat interface](screenshots/main.png)

## How it works

1. A PDF is uploaded, parsed into text, and split into chunks by the parser
   (so the chunking policy can't drift per format).
2. Chunks are embedded and stored in Qdrant; document metadata lives in
   Postgres.
3. A question is embedded, the nearest chunks are retrieved, and the LLM's
   answer streams back over SSE as it is generated.
4. The finished answer is persisted to Postgres together with the sources
   that were used, so reloading a document restores the full conversation.

## Architecture

```
┌──────────────┐  Vite dev   ┌───────────────────────────────────────┐
│   Frontend   │────────────▶│              Backend (Flask)          │
└──────────────┘             │                                       │
                             │  settings ── single global app_settings│
                             │  chat ── SSE stream, answers + sources│
                             │  eval ── retrieval/answer evaluator   │
                             │                                       │
                              │  parser (pymupdf fast / Docling)      │
                              │  storage (LocalStorage impl)          │
                             └──────┬──────────────┬───────────┬─────┘
                                    │              │           │
                             ┌──────▼─────┐ ┌──────▼────┐ ┌────▼─────┐
                             │  Postgres  │ │ Qdrant    │ │ Chat LLM │
                             │ (metadata, │ │ (vectors, │ │ (Google  │
                             │  messages, │ │  only)    │ │ or Groq, │
                             │  app       │ │           │ │  single  │
                             │  settings) │ │           │ │  key)    │
                             └────────────┘ └───────────┘ └──────────┘
```

## Design decisions

**Portable boundaries.** Two modules isolate vendor code so no single vendor is load-bearing.
The storage module (`backend/storage.py`) abstracts where uploaded files live;
the app currently ships the `LocalStorage` implementation, and anything that
can save, open, and serve a file can be substituted without touching route
code. The document parser (`backend/services/parsing/document_parser.py`) maps file
extensions to parsers; chunking is owned by the parser (token-based
`CHUNK_SIZE_TOKENS=512` / `CHUNK_OVERLAP_TOKENS=50` via `tiktoken
cl100k_base`) so swapping parsers cannot silently change chunk sizes.
Honest note: PDF has two branches behind the same parser — `pymupdf` fast path
default for born-digital single-column PDFs, and an opt-in Docling branch
(`USE_DOCLING=auto|true`, `.[docling]` extra, `granite-docling-258M` ~1.1GB)
that preserves tables as Markdown and reading order for two-column / scanned
/ borderless-table PDFs. The heuristic in `backend/services/parsing/pdf_heuristics.py`
routes only those PDFs to Docling; everything else stays on `pymupdf`.

**Single-instance, no auth.** The app has no users, no JWT, and no login
screen — every endpoint is open. A single `app_settings` row (provider, model,
`encrypted_api_key`) holds the BYO key, encrypted with Fernet derived from
`APP_SECRET` (see [docs/adr/0005-single-instance-no-auth.md](docs/adr/0005-single-instance-no-auth.md)).
`APP_SECRET` is optional locally but should be set in any persistent
deployment; changing it invalidates previously stored keys.

**Document title from context.** Storage keeps the uuid hex `filename` for
stable Qdrant and filesystem paths; the display `title` is derived from the
file's own context (PDF metadata Title → original filename → first heading)
and surfaced in the library, reader toolbar, and metadata (see
[docs/adr/0006-document-title-derived-from-context.md](docs/adr/0006-document-title-derived-from-context.md)).

**Streaming with persisted sources.** Answers stream token by token over SSE
rather than arriving as one block, because a retrieval answer can take long
enough to generate that blocking feels broken. Sources are attached when the
stream completes and stored alongside the answer, so the conversation survives
a reload. If the chosen provider fails mid-stream, the backend does not
silently answer through a different one; it surfaces the failure so the user
can fix their own key or quota. That no-fallback rule is deliberate — with
a single BYO key, a silent switch would hide the billing owner's error
(see [docs/adr/0001-per-user-byo-provider-keys.md](docs/adr/0001-per-user-byo-provider-keys.md)).

**Single-instance BYO keys.** Provider, model, and API key are global app
settings configured in the Settings dialog, not server environment variables.
Keys are encrypted at rest with Fernet and only ever returned masked. The app
boots with no provider key present; a workspace with no saved settings gets
a clear error pointing at Settings rather than a crash or an env default.

**Retrieval evaluation.** `backend/evaluation/` measures the retrieval
pipeline against a committed ground-truth fixture: ten questions over two
sample documents, scored with hit-rate and recall@k, plus an optional
LLM-as-judge faithfulness check on generated answers. The scoring logic runs
in tests against deterministic fakes; a live run against real providers is
opt-in because it costs API calls. This exists so changes to chunking or
retrieval can be judged with numbers instead of vibes.

## Testing

Run the fast suite against fakes and in-memory SQLite:

```bash
cd backend
uv run pytest
```

Run the full HTTP workflow against disposable Postgres and Qdrant services:

```bash
cd backend
./run-full-stack-tests.sh
```

The full-stack run applies every migration to an empty database, uses deterministic local embedding, reranking, and chat providers, and removes its database, collection, and containers when it finishes. It needs Docker but no model API keys or paid services.

The evaluator (`backend/evaluation/`) measures retrieval against
`fixture.json` with `hit@5`/`recall@5` (k=5) + per-question breakdown and
`ingest sec/PDF` (parse/embed/upsert wall time) via
`evaluation/evaluator.py`; live runs are opt-in (`--live`). See
[backend/README.md](backend/README.md) for free local live instructions
(`http://localhost:6333` without any chat key).

## Technologies

- Backend: Python, Flask, SQLAlchemy, Alembic
- Frontend: React, TypeScript, Vite, Tailwind CSS, Zustand, React Query
- Database: Postgres
- Vector store: Qdrant (local, `http://localhost:6333`)
- Embeddings: BGE-M3 local via `sentence-transformers` (CPU, 1024d, no key)
- LLM: Google Gemini or Groq, single-instance via the Settings dialog (BYO key, encrypted at rest)
- Chunking: `tiktoken` `cl100k_base`, `CHUNK_SIZE_TOKENS=512` / `CHUNK_OVERLAP_TOKENS=50`
- Retrieval: named dense vectors plus hashed term-frequency sparse vectors with Qdrant IDF and RRF, gated local cross-encoder reranker
