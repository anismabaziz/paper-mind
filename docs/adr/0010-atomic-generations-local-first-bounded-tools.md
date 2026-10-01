# Atomic generations, local-first operation, bounded tools

Chosen: three decisions that are expensive to reverse, so they are recorded
together while changing any one of them still means rebuilding the rest.

## Atomic Index Generation activation

A reindex builds a complete generation beside the active one and flips to it
only after validation. The alternative — updating the live index in place —
would leave every crash, timeout, or partial upsert as a Document whose
vectors half belong to two configurations, and chat would answer from the
mixture without knowing. The price is storage for two generations during a
reindex and a cleanup path that must itself be durable, which is why
deletion and generation cleanup share the same retryable machinery.

## Local-first operation

Embeddings, sparse indexing, reranking, Postgres, Qdrant, and the trace log
all run on the operator's machine; only the chat completion and the judge
calls leave it. That keeps the demo free, the evaluation reproducible, and
Document text out of any pipeline except the provider the reader chose. The
price is a heavier install (Docker, model downloads, CPU indexing measured
in the evaluation reports) and no hosted story: there is no multi-user
account, no auth, and no deployment target in this repository.

## Bounded tools for the Research Brief

The brief's model loop may call four read-only tools under fixed ceilings
(2 Documents, 6 turns, 8 tool calls, 2 repeated calls, 120,000 tokens,
120 seconds). An unbounded agent loop would be a second product — with its
own planning, permissions, and cost story — and this repository is not that.
The price is scope: cross-Document questions wider than a pair, and any tool
that writes, browses, or executes, are refused rather than approximated.

Rejected: in-place reindexing with a "reindex" flag on the Document. A flag
cannot distinguish a Document that failed halfway from one that was never
indexed, so every reader of the flag reimplements the generation check badly.

Rejected: server-side embeddings or a hosted vector store as the default.
Either would make the free local path the special case and put Document
text on someone else's machine before the reader picks a provider.

Rejected: a general tool sandbox for the brief. Sandboxing an executor the
product does not need would trade a reviewed refusal for an unreviewed
permission boundary.
