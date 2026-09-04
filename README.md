# PaperMind

Chat with your research papers. Upload a PDF, wait for it to index, then ask
questions and get answers streamed in from an LLM, each grounded in the
retrieved chunks of your document with the sources shown inline.

This is a portfolio project built to run locally. It is PDF-only by design:
the parser seam currently registers exactly one parser, for `.pdf`. There is
no hosted deployment and no multi-tenant story. What it does, it does on your
machine.

## Features

- PDF upload, parsing, chunking, and indexing into a vector store
- Streaming chat over the indexed document (SSE), with retrieved sources
  attached to each answer
- JWT auth with a demo mode for frictionless local use
- Retrieval evaluator with a committed ground-truth fixture
- Postgres for persistence, local filesystem for uploaded files

## Quick start

You need Docker and API keys for Pinecone plus one chat provider (Google or
Groq). From a clean clone:

```bash
export PINECONE_API_KEY=...        # vector store
export GOOGLE_API_KEY=...          # or GROQ_API_KEY, see MODE below
export DEMO_MODE=true              # skip login, good for a first look

docker compose up --build
```

That boots Postgres, runs the migrations, and serves the API on
`http://127.0.0.1:3000` (`GET /health` to check). The frontend is a Vite app
and runs separately:

```bash
cd frontend
npm install
npm run dev
```

Then open the printed localhost URL. Environment variables for the frontend
are documented in [frontend/.env.example](frontend/.env.example), and the
backend's in [backend/.env.example](backend/.env.example).

If you'd rather run the backend without Docker, the manual path (uv, local
Postgres, Alembic) is in [backend/README.md](backend/README.md).

## Screenshots

![Main chat interface](screenshots/main.png)

## How it works

1. A PDF is uploaded, parsed into text, and split into chunks owned by the
   parser seam (so the chunking policy can't drift per format).
2. Chunks are embedded and stored in Pinecone; document metadata lives in
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
                             │  chat ── SSE stream, answers + sources│
                             │  eval ── retrieval/answer evaluator   │
                             │                                       │
                             │  parser seam (pluggable per format)   │
                             │  storage seam (LocalStorage impl)     │
                             └──────┬──────────────┬───────────┬─────┘
                                    │              │           │
                             ┌──────▼─────┐ ┌──────▼────┐ ┌────▼─────┐
                             │  Postgres  │ │ Pinecone  │ │ LLM API  │
                             │ (metadata, │ │ (vectors) │ │ (Google  │
                             │  messages) │ │           │ │ or Groq) │
                             └────────────┘ └───────────┘ └──────────┘
```

## Design decisions

**Portability seams.** Two boundaries exist so no single vendor is load-bearing.
The storage seam (`backend/storage.py`) abstracts where uploaded files live;
the app currently ships the `LocalStorage` implementation, and anything that
can save, open, and serve a file can be substituted without touching route
code. The parser seam (`backend/services/document_parser.py`) maps file
extensions to parsers; PDF is the only one registered today, and adding
another format means adding a parser class, not rewriting the ingest path.
Chunking lives in the seam itself so swapping parsers cannot silently change
chunk sizes.

**Vendor-neutral auth.** Auth is plain JWT with bcrypt-hashed passwords,
implemented in `backend/services/auth_service.py`. `DEMO_MODE=true` disables
the checks entirely, which keeps the app usable for a demo or a code review
without handing out accounts. Token signing falls back to a per-process
random secret when `JWT_SECRET` is unset, which is fine for a laptop and
documented as not fine for anything shared.

**Streaming with persisted sources.** Answers stream token by token over SSE
rather than arriving as one block, because a retrieval answer can take long
enough to generate that blocking feels broken. Sources are attached when the
stream completes and stored alongside the answer, so the conversation survives
a reload. If the primary provider fails mid-stream, the backend does not
replay the partial answer through the fallback; it surfaces the failure
instead.

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

Tests run against fakes and in-memory sqlite; they never touch real Pinecone,
the LLM, or real Postgres. The evaluator's live mode is documented in
[backend/README.md](backend/README.md).

## Technologies

- Backend: Python, Flask, SQLAlchemy, Alembic
- Frontend: React, TypeScript, Vite, Tailwind CSS, Zustand, React Query
- Database: Postgres
- Vector store: Pinecone
- LLM/embeddings: Google Gemini or Groq (chosen with `MODE`)
