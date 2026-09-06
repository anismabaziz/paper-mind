# PaperMind

Chat with your research papers. Upload a PDF, wait for it to index, then ask
questions and get answers streamed in from an LLM, each grounded in the
retrieved chunks of your document with the sources shown inline.

This is a portfolio project built to run locally. It is PDF-only by design:
the parser currently registers exactly one parser, for `.pdf`. There is
no hosted deployment and no multi-tenant story. What it does, it does on your
machine.

## Features

- PDF upload, parsing, chunking, and indexing into a vector store
- Streaming chat over the indexed document (SSE), with retrieved sources
  attached to each answer
- Per-user chat settings: bring your own provider (Google or Groq), model, and
  API key via the Settings dialog — stored encrypted, never returned in
  plaintext
- JWT auth with a demo mode for frictionless local use
- Retrieval evaluator with a committed ground-truth fixture
- Postgres for persistence, local filesystem for uploaded files

## Quick start

Free local path (no keys) — default:

```bash
# Start infra (Postgres + Qdrant) — backend is not a compose service
docker compose -f backend/compose.yaml up -d
# Run backend locally
cd backend
uv sync
uv run alembic upgrade head
DEMO_MODE=true uv run python app.py   # API on http://127.0.0.1:3000 (GET /health)
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

## Configuring a chat provider

There are no provider keys in the environment. Open the Settings dialog in the
app, pick a provider (Google or Groq) and a model from the curated list, paste
your own API key, and hit "Test connection" to verify before saving. Keys are
stored encrypted per user and applied to new chats immediately. In demo mode
the dialog configures the seeded demo user, so a recruiter can run the whole
flow without an account — the only thing they still need is one API key of
their own.

Everything below runs free and local — no API keys anywhere in the pipeline
except the chat LLM, which each user configures in the app:

| Concern | Detail |
|---|---|
| Vector store | Qdrant on `http://localhost:6333` (compose `qdrant` service, volume `qdrant_storage`), collection `pdf-index` |
| Embeddings | `BAAI/bge-m3` via `sentence-transformers`, CPU, 8192 ctx, 1024d Matryoshka, cached to `hf_cache` volume — no API key |
| Retrieval | Hybrid dense + BM25 sparse fused with `RRF(k=60)`, 50 candidates → 5, gated reranker `RERANK=true` (22M MiniLM ~10ms/50 or `bge-reranker-v2-m3` ~80ms/50) |
| Chunking | `CHUNK_SIZE_TOKENS=512` / `CHUNK_OVERLAP_TOKENS=50` (~10%) via `tiktoken cl100k_base`, per-page, `page_no` + `content_hash` metadata |
| Parser | `pymupdf` fast path default; `USE_DOCLING=auto` routes only image-only / borderless-table / 2-col PDFs to Docling (opt-in `.[docling]`), `USE_DOCLING=true` forces all |
| Chat LLM | Per-user Settings: provider (Google or Groq), curated model, your own API key — encrypted at rest |
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
                             │  auth ── JWT (bcrypt) or demo bypass  │
                             │  settings ── per-user provider config │
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
                             │  settings) │ │           │ │ per user)│
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

**Vendor-neutral auth.** Auth is plain JWT with bcrypt-hashed passwords,
implemented in `backend/services/accounts/auth_service.py`. `DEMO_MODE=true` disables
the checks entirely, which keeps the app usable for a demo or a code review
without handing out accounts. Token signing falls back to a per-process
random secret when `JWT_SECRET` is unset, which is fine for a laptop and
documented as not fine for anything shared.

**Streaming with persisted sources.** Answers stream token by token over SSE
rather than arriving as one block, because a retrieval answer can take long
enough to generate that blocking feels broken. Sources are attached when the
stream completes and stored alongside the answer, so the conversation survives
a reload. If the chosen provider fails mid-stream, the backend does not
silently answer through a different one; it surfaces the failure so the user
can fix their own key or quota. That no-fallback rule is deliberate — with
user-supplied keys, a silent switch would bill someone else's account
(see [docs/adr/0001-per-user-byo-provider-keys.md](docs/adr/0001-per-user-byo-provider-keys.md)).

**Per-user BYO keys.** Provider, model, and API key are per-user settings
configured in the Settings dialog, not server environment variables. Keys are
encrypted at rest with Fernet and only ever returned masked. The app boots
with no provider key present; a user who has not saved settings gets a clear
error pointing at Settings rather than a crash or an env default. Demo mode
maps anonymous requests to a seeded demo user whose settings are edited
through the same dialog.

**Retrieval evaluation.** `backend/evaluation/` measures the retrieval
pipeline against a committed ground-truth fixture: ten questions over two
sample documents, scored with hit-rate and recall@k, plus an optional
LLM-as-judge faithfulness check on generated answers. The scoring logic runs
in tests against deterministic fakes; a live run against real providers is
opt-in because it costs API calls. This exists so changes to chunking or
retrieval can be judged with numbers instead of vibes.

## Testing

```bash
cd backend
uv run pytest
```

Tests run against fakes and in-memory sqlite; they never touch real Qdrant,
the LLM, or real Postgres (heavy models mocked or `pytest.importorskip`'d; `uv run pytest` stays headless).

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
- LLM: Google Gemini or Groq, per user via the Settings dialog (BYO key, encrypted at rest)
- Chunking: `tiktoken` `cl100k_base`, `CHUNK_SIZE_TOKENS=512` / `CHUNK_OVERLAP_TOKENS=50`
- Retrieval: hybrid dense + BM25 (`rank-bm25`) with RRF, gated local cross-encoder reranker
