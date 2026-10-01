"""
Whether one Document's passages can be read right now.

Chat and the Research Brief both refuse a Document whose vectors are going away
or no longer say what the app serves, and they refuse it for the same reasons: a
reindex in progress is being rewritten under the reader, and a stale index was
built with a parser, model, or collection schema this build does not serve, so
querying it answers from an incompatible index. A Document that was never indexed
has nothing to query either.

The judgement lives here so the two paths cannot drift into disagreeing about
whether a Document is readable. What differs is only how each one says it, which
is why the refusal carries its own message: chat talks about asking a question
about it, a brief about including it in a pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from services.deletion import is_deleting_record
from services.indexing.state import REINDEXING_STATES, IndexState, index_status

#: Why a Document cannot be read. Named rather than worded so the callers can
#: say it in their own terms and both are still told the same thing happened.
UNREADABLE_MISSING = "file_not_found"
UNREADABLE_DELETING = "document_deleting"
UNREADABLE_INDEXING = "document_indexing"
UNREADABLE_STALE = "index_stale"
UNREADABLE_PENDING = "index_pending"


@dataclass(frozen=True)
class DocumentRefusal:
    """One Document that cannot be read, and what is wrong with it."""

    status: int
    category: str
    #: What a client is told alongside the refusal: the reindex action and the
    #: index state for a stale Document, the running job for one being reindexed.
    detail: dict[str, Any] = field(default_factory=dict)


def document_refusal(
    repositories: Any,
    settings: Any,
    filename: str,
) -> DocumentRefusal | None:
    """
    Return why one Document cannot be read, or None when it can.

    The index state is judged and recorded here rather than at each call site, so
    the staleness reason stored against the Document is the same whichever path
    asked.
    """
    file_record = repositories.files.get_file(filename)
    if not file_record:
        return DocumentRefusal(404, UNREADABLE_MISSING)
    if is_deleting_record(file_record):
        return DocumentRefusal(
            409, UNREADABLE_DELETING, {"deletion": file_record.get("deletion_state")}
        )
    ingestion_job = repositories.ingestion_jobs.get_latest(filename)
    if (
        file_record.get("is_processed")
        and ingestion_job
        and ingestion_job["state"] in REINDEXING_STATES
    ):
        return DocumentRefusal(409, UNREADABLE_INDEXING, {"job": ingestion_job})
    state: IndexState = index_status(repositories.files, file_record, settings)
    detail = {"action": "reindex", "index": state.to_dict(settings)}
    if state.is_stale:
        return DocumentRefusal(409, UNREADABLE_STALE, detail)
    if state.state == "pending":
        return DocumentRefusal(409, UNREADABLE_PENDING, detail)
    return None


#: What each refusal reads like when chat asks about a Document. A Document
#: being removed is not here: its refusal carries the deletion payload's own
#: wording, which names the retry rather than the block.
CHAT_REFUSAL_MESSAGES = {
    UNREADABLE_MISSING: (404, "File not found"),
    UNREADABLE_INDEXING: (
        409,
        "This document is being reindexed. Try again when indexing finishes.",
    ),
    UNREADABLE_STALE: (
        409,
        "This document's index no longer matches the current settings. "
        "Reindex it to ask questions again.",
    ),
    UNREADABLE_PENDING: (
        409,
        "This document is not indexed yet. Index it to ask questions.",
    ),
}

#: What each refusal reads like when a brief picks a pair. A brief names the
#: label because that is how the reader chose it, and a refusal that said "this
#: document" without saying which would leave them guessing.
BRIEF_REFUSAL_MESSAGES = {
    UNREADABLE_MISSING: (404, "Document {label} was not found."),
    UNREADABLE_DELETING: (
        409,
        "Document {label} is being removed, so it cannot be part of a brief.",
    ),
    UNREADABLE_INDEXING: (
        409,
        "Document {label} is being reindexed. Try again when indexing finishes.",
    ),
    UNREADABLE_STALE: (
        409,
        "Document {label}'s index no longer matches the current settings. "
        "Reindex it before including it in a brief.",
    ),
    UNREADABLE_PENDING: (
        409,
        "Document {label} is not indexed yet. Index it before including it in a brief.",
    ),
}
