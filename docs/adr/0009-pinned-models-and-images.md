# Immutable local model revisions, and no code from model repositories

Chosen: the local embedding model and the local reranker each load at a commit
recorded in `services/models.py` rather than at whatever the repository's
default branch points at. Both load through stock `transformers` architectures,
so `trust_remote_code` is off and there is no path that runs Python from a model
repository. A model that genuinely needs its own modelling code is refused
unless the exact commit was read and recorded in `REVIEWED_REMOTE_CODE_MODELS`
first. Postgres and Qdrant are pinned by digest; the committed lockfiles are
installed with `uv sync --frozen` and `npm ci`.

The reasoning. A model named by repository id is a moving target: the same
`BAAI/bge-m3` string resolves to different weights after an upstream push, and
nothing in a running application says so. The consequences are not subtle and
not confined to one subsystem. Retrieval quality shifts because the vectors
changed. The index becomes internally inconsistent, because documents indexed
before the push and after it hold vectors from different models in the same
collection. A published evaluation number stops describing the system, because
the report recorded a model id and no revision, so two runs that both say
`BAAI/bge-m3` cannot be compared. And the failure is silent: no test fails, no
log line changes, no health check trips.

The index manifest and the evaluation report both already had a revision field
that nothing ever wrote. Filling those fields from the resolved commit rather
than the raw environment variable is what makes the rest work. A document
indexed under one commit and queried under another is detected as stale and
asks for a reindex instead of returning confidently wrong neighbours. Two
evaluation reports become comparable because each names the weights it ran
against.

`trust_remote_code` was previously hardcoded to `True` in both loaders, with a
comment claiming BGE-M3 required it. It does not, and it could not have been
doing anything: at the pinned commits neither repository contains a single
`.py` file and neither config declares an `auto_map`, so there is no repository
code for the flag to permit. BGE-M3 is an XLM-RoBERTa encoder with a standard
sentence-transformers module list, and the MiniLM cross-encoder is a standard
BERT sequence classifier. Executing Python from the model repository bought
nothing and would have meant running unreviewed code fetched over the network
during a document upload.

Rejected: leaving the revision unset and treating the field as documentation.
That is what was there before, and it is indistinguishable from a pin in every
artefact a reader consults — a manifest saying `""` reads like a deliberate
"no revision" rather than "nobody checked".

Rejected: pinning only the digest of the weight file. A file hash proves which
bytes arrived, but not that the tokenizer, the pooling configuration, and the
architecture come from the same reviewed state. A commit covers the repository.

Rejected: hashing the resolved revision into a local file at first download.
That makes the *first* download the reference, which is exactly the unpinned
behaviour being removed, and it silently accepts whatever was cached on a
developer's machine.

Also rejected: a floating tag with an integrity check on the tarball. It
protects the download and not the selection, so a retagged branch still changes
the weights.

## Application and worker images

The ticket asked for the Postgres, Qdrant, application, and worker images all to
be pinned. Three of those four exist. There is no application image and no
worker image: the repo has no Dockerfile, and both the Flask app and the
ingestion worker run from a host virtualenv (`uv run python app.py`,
`uv run python worker.py`). The `backend/.dockerignore` is a leftover from an
earlier container layout and nothing consumes it.

So there is nothing to pin for the app and the worker, and the honest
resolution is to record that rather than to invent images in order to satisfy
the requirement. A container for the application and the worker would be
worth building for deployment, and the moment that happens the pins come with
it: a built image is pinned by its own digest, and the base image underneath
it by the digest recorded here. The `python-version` that CI installs is pinned
to 3.11, the floor in `requires-python`, and the local model weights those
processes load are pinned above, so the versions that decide behaviour are
recorded even without a container boundary.

The consequence of not containerising is that `uv.lock` is the only thing
standing between a checkout and the versions it runs, which is exactly why CI
installs it with `--frozen` and why the audit is a required check rather than a
suggestion.

Trade-off: moving a pin forward is a deliberate act — resolve the new commit,
record it, and the change marks every indexed document stale, which is a
reindex. That is the intended cost. A silent upgrade would be cheaper right up
until the day it was not, and the report a reader is asked to trust would be
describing weights that no longer exist.

A repository id that Hugging Face redirects is resolved before the pin table is
consulted. The default reranker setting uses
`cross-encoder/ms-marco-MiniLM-L-6-v2`, which redirects to
`cross-encoder/ms-marco-MiniLM-L6-v2`; the API answers only for the canonical
spelling, so a pin keyed by the alias could never be checked and a pin keyed
only by the canonical form would be lost the moment an operator wrote the
default spelling. The revision travels with the weights either way, and the
library still receives the id the operator wrote, so the Hugging Face cache
directory stays the one they can recognise and delete.

Hard to reverse: documents indexed under the old commit stay stale until they
are reindexed. The stored vectors are not migrated, because there is no
correct way to bring vectors from one model into another — they are recomputed
on ingest. The pin is a commit, so moving forward is not blocked by having
moved before.
