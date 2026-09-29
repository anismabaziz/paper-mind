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
- Traces: every answer request writes one redacted trace to `backend/data/traces/answer-traces.jsonl`. It holds Document, Conversation, Turn, and index generation IDs, the retrieval ranking (method, candidate/fused ranks, rerank result, filter, candidate count, latency), the generation step (prompt version, provider, model, token counts, time to first token, latency, finish reason, retry count, cost), and the citation and persistence outcomes. Questions, answers, prompts, Document text, and API keys are stored as fingerprints, never as text. Set `TELEMETRY_OTLP_ENDPOINT` to also send the same redacted spans to a collector. Set `TELEMETRY_CAPTURE_CONTENT=true` to write the words into the local file instead; the window closes on its own after `TELEMETRY_CAPTURE_WINDOW_SECONDS`, and credentials are stripped either way.
- All knobs are documented in `.env.example`.

## Setup (one command, from a clean clone)

```bash
# from the repo root — generate local-only secrets once, then start infra
backend/scripts/bootstrap-local.sh
docker compose -f backend/compose.yaml up -d
# then run backend locally
cd backend
uv sync
uv run alembic upgrade head
uv run python app.py   # API on http://127.0.0.1:3000 (GET /health)
```

`bootstrap-local.sh` is idempotent: it creates gitignored
`backend/.infra.env` with a random Postgres password and writes a matching
`DATABASE_URL` into `backend/.env`. No step edits a published port — every
service binds `127.0.0.1` by default, so the database and the vector store
are reachable from your machine and invisible to the LAN.

The only required env vars are `DATABASE_URL` (written by the bootstrap
script) and `QDRANT_URL` (defaults to `http://localhost:6333`). Optional:
`QDRANT_API_KEY` (empty for loopback development; required before any remote
address) and `APP_SECRET` — the Fernet root that
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
traceback. A remote `QDRANT_URL` without `QDRANT_API_KEY` is also refused at
boot with the same readable error.

## Local network and secrets

PaperMind is a single-user local app. The threat model is simple: your PDFs,
their vectors, and the database credentials that guard them must never reach
another machine on the network unless you explicitly allow it.

- Postgres (`5432`), Qdrant HTTP (`6333`), and Qdrant gRPC (`6334`) publish
  on `127.0.0.1` only, in both `compose.yaml` and `compose.test.yaml`.
- The Postgres password is generated per machine by
  `backend/scripts/bootstrap-local.sh` into gitignored `backend/.infra.env`
  — no fixed default ships in source control.
- Qdrant runs without an API key on loopback. The backend refuses to start
  against a non-loopback `QDRANT_URL` unless `QDRANT_API_KEY` is set, and
  `compose.qdrant-auth.yaml` is the compose path that turns the key on for the
  server. It is an overlay rather than a line in `compose.yaml` because even an
  empty `QDRANT__SERVICE__API_KEY` switches Qdrant into auth-required mode and
  would lock out keyless local clients.
- Development and test credentials are isolated: the full-stack suite mints
  an ephemeral `POSTGRES_TEST_*` password per run and never reads
  `backend/.infra.env`.
- `tests/test_local_network.py` proves the effective published addresses (both
  the compose file and `docker compose config`), the remote-without-key refusal,
  and what the bootstrap script generates.

Before exposing anything beyond loopback:

```bash
export QDRANT_API_KEY='<strong unique value>'
docker compose -f backend/compose.yaml -f backend/compose.qdrant-auth.yaml up -d
```

Put the same value in `backend/.env`, front the service with TLS, use a strong
unique Postgres password, and confirm the rendered bindings with
`docker compose -f backend/compose.yaml config` — no line may show `0.0.0.0`.

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

`evaluation/` asks a versioned labeled case set
(`evaluation/datasets/`): 57 reviewed questions over four sample documents in
`evaluation/sample_docs/` — three authored in-repo (CC0), one published paper
(CC BY 4.0). Every case goes through the application's answer path, the one
`POST /response` uses, so the numbers describe the app rather than a second
implementation of it. The sample documents are stored and indexed by the same
ingestion job and worker the upload route uses, which gives each one a real
Conversation, a real index generation, and a real index manifest, and the run
reports how long each took to index.

Each case declares what it expects: the outcome the run has to end as, the
passages that have to support the answer, and a rubric saying what a correct
answer has to contain, which reaches the judge so a reworded answer is not
scored against one reference wording. The set is split into a tuning half and a
reported half before any configuration was chosen, so a quoted number was not a
fitted one, and each source document is pinned by content hash, so a replaced
file cannot quietly keep answering to expectations reviewed against the old
one. A question the document cannot answer may be abstained on or declined, and
both are accepted; the cases that need a failure injected are held back from a
run and named in the run record.

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
uv run python -m evaluation.cli --live --split tuning      # the tuning half instead of the reported one
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

### Reports and ablations

A number on its own is a claim, so `--report` writes what a reviewer can check
beside it. The directory holds a `manifest.json` — the revision, the case set
version and split, every document's content hash, every document's index
manifest and generation, the prompt version, the models and their revisions,
the settings, and the environment, including the retrieval implementation's own
versions — one `results/<experiment>.json` per experiment, and a `README.md`
summary with the table, the questions that moved, and what the run did not
measure.

The experiments themselves are declared in `evaluation/experiments.py`, one
question each: dense only, sparse only, hybrid only, reranking on, candidate
depth 10 and 100, query expansion off and on, and two chunking policies. Every
value was chosen on the tuning split, and the loader refuses an experiment that
claims to have been chosen on the split the result is quoted from. A retrieval
variant is measured against the index that already exists; a chunking variant
indexes the documents again, because different chunks are different vectors.

```bash
uv run python -m evaluation.cli --live --ablate --report evaluation/reports/<name>
```

`--ablate` measures retrieval only, so it needs no key and reports no answer,
citation, abstention, token, or dollar figure — the manifest says so rather than
leaving those cells looking empty. Two reports are published, both from real
documents, real BGE-M3 embeddings, and real Qdrant:
`reports/2026-09-retrieval-baseline-v1/` over the reported half of the case set
and `reports/2026-09-retrieval-tuning-v1/` over the tuning half, which is the
half every variant's value was chosen on. Read the tuning report for why a
variant exists and the baseline report for what it does.

Re-running says whether the run reproduced the published one:

```bash
uv run python -m evaluation.cli --live --ablate \
  --report /tmp/rerun --compare-report evaluation/reports/2026-09-retrieval-baseline-v1
```

Timing and dollars are deliberately outside the manifest's digest, because they
move on every run; what a reproduction is judged on is the retrieval numbers,
the outcomes, the verdicts, and a changed model revision, which is reported on
its own since a checkout cannot pin that.

`--pace SECONDS` is the wait between provider calls, six seconds by default.
A rate-limited account measures its allowance in tokens per day, and a run that
asks for everything at once spends it and then reports the cases it could not
afford as provider failures — a number about the account, not about the
application. `--pace 0` removes the wait for a paid account. A judged run over
the reported half costs roughly 70k generator tokens and 120k judge tokens, so
on a free daily allowance it is about one run per model per day. The two
allowances are separate: the generator's key and the judge's key each get their
own budget, which is why the run configures them apart.

Reasoning models spend their output budget thinking before they answer, and an
empty reply is what a budget that ran out looks like. The judge runs under a
4k output budget for exactly this reason, while the app's own answers keep the
catalog's 1k; a judge verdict that never arrives is reported Unknown, never
scored as a zero.
