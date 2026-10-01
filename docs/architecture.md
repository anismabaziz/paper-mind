# Architecture

PaperMind is a local-first workspace for chatting with research papers. One
Document holds one Conversation, and every exchange in that Conversation is
one ordered Turn. This page names the moving parts with the project's own
words — the project's shared glossary — so a reader can trace a PDF from
upload to cited answer without guessing.

## Data stores

Three stores hold three kinds of state, and nothing else pretends to be one
of them:

- **Postgres** — metadata, Conversations, Turns, Citation Sources, the single
  `app_settings` row, Ingestion Jobs, and the Index Manifest of every
  Document. It is the only store migrations touch.
- **Qdrant** (`pdf-index` collection) — vectors only. Each point carries a
  named 1,024-dimensional dense vector and a hashed term-frequency sparse
  vector for one Passage of one Index Generation, plus `page_no`,
  `chunk_index`, and `content_hash` payloads. Document text lives in Postgres
  and on disk, never as the source of truth in Qdrant.
- **Filesystem** — uploaded PDFs under stable uuid filenames, the trace log
  at `backend/data/traces/answer-traces.jsonl`, and the local model cache.

There are no cross-store transactions. Postgres job state, Index Generation
activation, and retryable cleanup are the consistency model.

## Index Generation and Index Manifest

Every Document's vectors belong to one numbered **Index Generation**. A
reindex builds the next generation beside the active one — new point IDs
derived from Document, generation, and Passage — and activates it only after
parsing, embedding, sparse indexing, and validation succeed. The generation
it replaced stays readable until a recorded cleanup removes it, so a failed
attempt never damages the working index. Retrieval reads the active
generation only; a generation being built is never observed.

The **Index Manifest** records what the active vectors were built with:
Document content hash, parser and version, chunk size and overlap, embedding
model and revision, vector dimension, sparse method and tokenizer version,
reranker model and revision, collection schema version, and the active
generation. When the running configuration no longer matches the manifest,
the Document is in **Stale Index** state: its vectors stay readable, chat
refuses with a reindex action, and reindexing queues an ordinary Ingestion
Job that replaces the generation only once the replacement validates.

## Ingestion Jobs

Uploads never index inline. Storing the PDF creates one queued **Ingestion
Job** in the same transaction and returns; a Postgres-backed worker claims
queued jobs and moves them through parsing, embedding, indexing, and
validation while refreshing a heartbeat and recording stage, progress,
attempt count, and timestamps. One Document has at most one active job, and
processing and deleting the same Document are serialized, so late work
cannot recreate data a deletion just removed.

A job is `queued`, `running`, `cancelling`, `cancelled`, `failed`, `ready`,
or `stale`. Failure carries a safe, actionable error category — never a
traceback — with a retry action. Deletion runs through the same machinery:
the Document enters a `deleting` state (a tombstone that blocks new chat and
ingestion work), vectors and file bytes are removed, and only then is the
metadata finalized. Partial failure keeps retryable cleanup state instead of
claiming success.

## Retrieval: typed outcomes, one fusion

A question is embedded once and searched as dense plus sparse in a single
Qdrant query, fused by `RRF(k=60)` over 50 candidates down to 5 Passages,
with an optional gated local cross-encoder rerank. Every call returns a
typed result: `success` or `empty`, the method that actually ran (`dense`,
`sparse`, or `hybrid`), the candidate ranks before and after fusion and
rerank, and the Index Generation that was read. A store that cannot serve
hybrid fails the call as `VectorStoreUnavailableError`,
`VectorStoreConfigurationError`, or `VectorDimensionError` — never as an
empty result, so an outage is never mistaken for a lack of relevant
Passages. Retrieval scores stay internal; the interface shows rank and
method, never a score dressed as a confidence percentage.

## Chat: one Turn, one terminal event

Each question is committed as a `pending` **Turn** with a stable sequence
number before the provider is called, so a provider failure, a persistence
failure, or a client that walks away all end as a recorded terminal state
(`answered`, `failed`, `cancelled`, or `unanswered`) rather than a stranded
question. The prompt carries a bounded window of recent Turns
(`CHAT_RECENT_TURNS=4`, transcript capped at 1,200 tokens, evidence at 6,000)
with deterministic query expansion for follow-ups.

Every Passage gets a stable source id (`S1`, `S2`, …) before generation, and
the model returns claims bound to those ids. Invalid ids are corrected once,
then end the Turn as a citation error rather than a false citation. When
retrieval leaves nothing usable, the app abstains before calling the paid
model and stores the abstention reason. The SSE stream opens with `start`,
carries `token` events, and ends with exactly one terminal event (`done`,
`abstained`, `provider_error`, `citation_error`, `persistence_error`, or
`cancelled`) — and `done` is sent only after the Turn and its Citation
Sources commit.

## Research Brief: a bounded pair with four tools

A Research Brief answers one question across exactly two chosen Documents
(the pair the trace records; a wider brief would make that field
meaningless). The model may call four read-only tools — library search,
Passage retrieval, Page text, and evidence comparison — under six ceilings:
2 Documents, 6 turns, 8 tool calls, 2 repeated calls, 120,000 tokens, and
120 seconds. The result is a structured brief with claims, supporting Page
citations, conflicting evidence, and explicit gaps. It cannot execute code,
browse the web, change App Settings, or delete Documents.

## Traces and metrics

Every answer writes one redacted trace: Document, Conversation, Turn, and
generation ids; retrieval method, candidate and fused ranks, and rerank
result; prompt version, model, token counts, time to first token, latency,
finish reason, retry count, and cost; citation validation and persistence
outcomes. Questions, answers, prompts, Document text, and keys are stored as
fingerprints, never as text, unless a time-bounded local debug window is
opened. The same redacted spans can go to an OTLP collector. Aggregated
metrics cover success rate, latency, cost, retrieval quality, citation
validity, and failure category — see the evaluation reports for the measured
values.
