# PaperMind Backend

Flask API for RAG chat over uploaded PDFs.

## Setup

```bash
cd backend
uv sync
cp .env.example .env   # then fill in the values
```

On boot the app validates its environment: any missing required variable
is named on stderr with a pointer to `.env.example`, instead of a library
traceback.

## Run

```bash
uv run python app.py
```

The API serves on `http://127.0.0.1:3000` (`GET /health` to check).

## Tests

```bash
uv run pytest
```

Tests never talk to real Pinecone, LLM, or Supabase services —
`tests/conftest.py` provides dummy environment values for every run.
