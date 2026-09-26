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
- Follow-up context: the last `CHAT_RECENT_TURNS=4` answered turns, with the transcript capped at `CHAT_PRIOR_TURNS_TOKEN_BUDGET=1200` tokens and the evidence at `CHAT_CONTEXT_TOKEN_BUDGET=6000`. Older turns and lower-ranked Citation Sources are dropped first; the current question is never truncated. Both drops are reported in the `retrieval` block of the terminal SSE event.
- `POST /response` streams one answer per request. Events: `start` (the recorded Turn), `token`, then exactly one terminal event — `done`, `abstained`, `provider_error` (with a `category` of `provider`, `timeout`, or `empty_output`), `persistence_error`, or `cancelled`. `done` is only sent after the Turn and its Citation Sources are committed. It carries `finish_reason` and `truncated`, which is `true` whenever the provider did not report a finish reason that means "the answer is whole" — including when the app's own answer cap stopped the stream. A stream that never sends a terminal event is a client-side failure, not a pending answer.
- `abstained` means retrieval left nothing to answer from, so no model was called and nothing was spent. It carries a `reason` of `no_evidence` (the index held no passage for the question) or `evidence_unusable` (it matched passages that could not be read or cited), plus the `message` the user reads. The Turn is stored with that reason and no Citation Sources. An unreachable vector store is not an abstention: it stays a 503, so an outage is never replayed as an answer.
- Each catalogued model carries its own chat budget: `max_input_tokens` (the app trims the prompt to fit it — no SDK takes it as a parameter), `answer_token_budget` (sent on every call), `generation_timeout_seconds` (a stream that stalls is abandoned at it), and the finish reasons that API reports. Stored answer text is additionally capped at `max_answer_chars`. A transient provider failure is retried once, on the same provider, and only before the first fragment is visible.
- Query expansion: a follow-up is searched with the recent user questions prepended, so "the second method" has terms to match. `CHAT_QUERY_REWRITE=false` (default) spends no extra model call; set it to `true` to rewrite with the model instead, which falls back to the deterministic expansion on failure. The rewrite runs before retrieval, because retrieval needs the query it produces, so with it enabled a follow-up spends that one call even when the answer turns out to be an abstention.
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
  provider → model capability catalog
- `PUT /settings` — verify the candidate, then encrypt and replace the single
  global settings row
- `POST /settings/verify` — verify candidate provider/model/key values without
  changing stored settings

The settings routes are open (no auth) and use the single `app_settings` row.
Chat runs on those global settings — a workspace with no saved settings gets
a clear "configure a provider in Settings" error, and the backend boots fine
with no keys at all. Evaluation reads its keys from
`PAPERMIND_EVAL_GENERATOR_API_KEY` and `PAPERMIND_EVAL_JUDGE_API_KEY` instead.

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

The API and the ingestion worker are separate processes. Both need the same
database and services:

```bash
uv run python app.py        # API on http://127.0.0.1:3000 (GET /health)
uv run python worker.py     # claims and processes ingestion jobs
```

Upload stores the PDF and creates one queued ingestion job in the same
transaction, then returns immediately. The worker claims queued jobs, moves
them through parsing, embedding, indexing, and validation while refreshing a
heartbeat, and records stage, progress, attempt count, resource usage, and a
safe error category. A worker that stops leaves a recoverable job. Users can
cancel queued or running work through the ingestion-job API; a cancelled or
limited job remains retryable. The worker checks these bounds before and after
each ingestion stage. Configure `MAX_INGESTION_PAGES`,
`MAX_INGESTION_TEXT_BYTES`, `MAX_INGESTION_OUTPUT_BYTES`,
`MAX_INGESTION_SECONDS`, and `MAX_INGESTION_MEMORY_BYTES` to bound work per
job. `uv run python worker.py --once` drains the current queue and exits.

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

`evaluation/` asks a committed labeled case set
(`evaluation/fixture.json`): ten questions over two sample documents in
`evaluation/sample_docs/` — one authored in-repo (CC0), one published paper
(CC BY 4.0). Every case goes through the application's answer path, the one
`POST /response` uses, so the numbers describe the app rather than a second
implementation of it. The sample documents are stored and indexed by the same
ingestion job and worker the upload route uses, which gives each one a real
Conversation, a real index generation, and a real index manifest, and the run
reports how long each took to index.

Each case ends as whatever it was: an answer, an abstention, a provider
failure, an unusable citation, a failed save, or a refusal. Only an answer the
model wrote reaches a grader — retrieved context quoted back after a
provider failure is a fallback, not an answer. Every run records the index
manifest and generation, the retrieval method, the prompt version, the
provider, the model, and the settings behind the numbers, and the retrieval
contract is checked once up front, so a store that cannot serve hybrid fails
the run rather than quietly reporting dense numbers.

### What a run reports

**Retrieval** (`evaluation/metrics.py`) scores each question four ways and
reports every question's row beside the aggregate: hit rate, recall, MRR for
the rank of the first relevant chunk, and nDCG. The ideal nDCG ranking is one
relevant chunk per gold snippet, so a run that found some of the evidence cannot
score a perfect ranking with what it did find — which is the difference between
this and hit rate, where half the evidence at rank 1 reads as a full hit. A
chunk counts once however many gold snippets it holds, so the score does not
move when the chunk size setting changes.

**Answers** (`evaluation/graders.py` and `evaluation/judge.py`) carry two kinds
of grader, and the split matters. Six graders decide by reading the run's own
record and never call a model: whether the case ended as the case set expected
(provider status), whether a question that had to be refused was refused and
one that could be answered was answered (abstention), whether the answer's own
citations reach the expected evidence, whether the stored claims are readable,
whether every claim names a supplied Passage, and whether every page the answer
points at is a page of a Passage that claim cites. Two more are fractions:
citation precision (does the cited Passage repeat the claim's own wording) and
citation recall (does each claim cite something at all).

The other two need a judge: faithfulness against the retrieved context, and
correctness against the case set's expected answer. The judge is a second
model, built through the app's own provider factory with its own provider,
model, and key, and it answers against a versioned rubric
(`faithfulness-rubric-v2`) that the run record names. A judge that returns no
verdict is reported as **Unknown**, never as unfaithful — a broken judge is not
evidence of a hallucinating model, and a reply that negates the word it names
("not faithful") is a sentence rather than a verdict, so it is Unknown too.
Every judged run also grades the hand-labelled set in
`evaluation/calibration.json` (faithful, partial, unfaithful, and one case a
person could not settle) and reports where the judge disagreed with the
person. `--no-calibration` skips it.

Every metric reports `graded`, `scored`, `mean`, `passed`, `failed`, and
`unknown` counts, so a case nothing could be decided about is visible rather
than averaged in as a zero. A provider failure fails the case: it never
improves a quality score, and a case that generated nothing is not priced.

**Latency and cost** come from the clock the run was given. Retrieval, time to
first token, and total latency are each reported at p50 and p95, and the first
case whose retrieval actually ran is held apart from the rest, because that is
the one paying for whatever loads lazily; a case refused before retrieval
measures nothing and is labelled as such rather than counted as warm. A run
also reports input and output tokens, how many answers were truncated, each
finish reason, and the cost estimated from the catalog's published prices, next
to the prices themselves. Input tokens count the system instruction every call
carries alongside the question, transcript, and evidence; output tokens count
the claims block the model wrote, which is stripped before the answer is stored
but was billed for. The token totals and the dollar figure count the same cases:
the ones that reached a model and answered.

`uv run pytest` runs the whole thing offline against deterministic embedding,
vector store, and provider doubles. No Qdrant, no LLM, no paid calls.

Live run, opt-in because it indexes real documents and (optionally) bills a
provider. Keys come from the environment, never from a command-line argument:

```bash
cd backend
# Free local path: Qdrant on http://localhost:6333, no chat key needed
# (requires: docker compose -f compose.yaml up -d qdrant, or QDRANT_URL=http://localhost:6333)
export PAPERMIND_EVAL_GENERATOR_API_KEY=...
uv run python -m evaluation.cli --live --no-judge          # retrieval and outcomes, no judge
uv run python -m evaluation.cli --live                     # + model-judged faithfulness and correctness
uv run python -m evaluation.cli --live --no-calibration    # judge without the labelled set
uv run python -m evaluation.cli --live --json              # machine-readable
uv run python -m evaluation.cli --live --rerank            # force RERANK=true (local cross-encoder 50→5)
uv run python -m evaluation.cli --live --compare-rerank    # the case set twice: gate off, then on
```

The generator and the judge are configured apart, so a run can generate with
one account and grade with another:

```bash
export PAPERMIND_EVAL_GENERATOR_API_KEY=...   # for --provider/--model
export PAPERMIND_EVAL_JUDGE_API_KEY=...       # for --judge-provider/--judge-model
uv run python -m evaluation.cli --live \
  --provider groq --model openai/gpt-oss-20b \
  --judge-provider google --judge-model gemini-2.5-flash
```

A live run stores the sample documents under an `eval-` prefix and deletes
them afterwards. The index lives at `http://localhost:6333` (compose exposes
6333→6333 and 6334→6334).
