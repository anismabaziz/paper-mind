# Known limits

What PaperMind does not do, stated plainly so no demo or screenshot implies
otherwise.

## Single-user local operation

One workspace, one set of App Settings, no login, no accounts, no sharing.
Every endpoint is open on `127.0.0.1`; Postgres, Qdrant HTTP, and Qdrant
gRPC bind to loopback by default, and the Postgres password is generated
per machine. Anything beyond loopback is unsupported without a Qdrant API
key, TLS in front, and strong unique secrets — see `backend/README.md` for
the threat model. There is no hosted deployment and no multi-user story.

## Provider-specific capabilities

Chat runs on Google or Groq through the reader's own key, configured in the
Settings dialog. Only catalogued models can be picked
(`gemini-2.5-flash`, `gemini-3.5-flash`, `openai/gpt-oss-120b`,
`openai/gpt-oss-20b`, `qwen/qwen3.8-27b`); a retired model is removed after
the scheduled availability check fails rather than left selectable. Research
Brief needs structured output plus tool use, so a model without both stays
available for document chat but cannot run a brief.

## Hosted-model data transfer

Local work — parsing, chunking, embeddings, sparse indexing, reranking —
never leaves the machine. Asking a question sends the selected Passages and
the bounded recent Turns to the configured provider's cloud, and judging an
evaluation run sends case text to the judge's provider. There is no
local-only question mode: a sensitive Document should be read, not asked
about, until its provider is acceptable.

## Not production-ready

- PDF only: the parser registers exactly one parser, for `.pdf`.
- Research Brief answers across exactly two Documents, not up to five, and
  cannot execute code, browse sites, change settings, or delete Documents.
- Retrieval scores are ranking internals; the interface shows rank and
  method, never a calibrated confidence.
- Long-document limits are enforced per job (`MAX_INGESTION_*`); a malformed
  PDF is refused rather than ingested partially.
- Evaluation numbers describe the case sets and models named in each report,
  not the application in general; differences smaller than a run's own
  movement are not differences.
