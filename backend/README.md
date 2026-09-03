# PaperMind Backend

Flask API for RAG chat over uploaded PDFs. Runs entirely on Postgres and
the local filesystem — no external SaaS beyond the LLM/vector providers.

## Setup (one command, from a clean clone)

```bash
# from the repo root; requires Docker and PINECONE_API_KEY (plus a
# GOOGLE_API_KEY or GROQ_API_KEY depending on MODE) in your shell or .env
docker compose up --build
```

This boots Postgres, runs the Alembic migrations, and serves the API on
`http://127.0.0.1:3000` (`GET /health` to check).

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
hit-rate/recall@k for retrieval and LLM-as-judge faithfulness for the
answers it generates.

The test suite exercises the evaluator's scoring logic on deterministic
fakes. A live run — real embeddings, real retrieval in Pinecone, real
answers, real judge — is opt-in because it costs provider calls:

```bash
cd backend
uv run python -m evaluation.cli --live              # full report
uv run python -m evaluation.cli --live --no-judge   # retrieval metrics only
uv run python -m evaluation.cli --live --json       # machine-readable
```

A live run indexes the sample docs under an `eval-` prefix in the vector
index and deletes them afterwards.

