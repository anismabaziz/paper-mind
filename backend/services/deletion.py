"""
Whether a Document can be asked a question right now, and what to say if not.

Both the HTTP route and the evaluator have to answer that question the same
way, so the verdict and its payload live here rather than in a route helper:
a Document that is being deleted, reindexed, or whose index no longer matches
the running configuration is refused with the same words wherever it was asked
from.
"""

from typing import Any

#: Deletion states that mean a Document is on its way out and must not be used.
DELETING_STATES = ("deleting", "delete_failed")


def is_deleting_record(file_record: dict | None) -> bool:
    """Return whether a document is mid-deletion or failed deletion."""
    if not file_record:
        return False
    return file_record.get("deletion_state") in DELETING_STATES


def deletion_block_payload(file_record: dict[str, Any] | None) -> dict[str, Any]:
    """Return the payload explaining that a Document is being deleted."""
    record = file_record or {"deletion_state": "deleting"}
    if record.get("deletion_state") == "delete_failed":
        return {
            "error": (
                "This document failed to delete. Retry deletion before using it again."
            ),
            "category": "document_delete_failed",
            "deletion_state": "delete_failed",
            "deletion_error": record.get("deletion_error"),
        }
    return {
        "error": "This document is being deleted.",
        "category": "document_deleting",
        "deletion_state": "deleting",
    }
