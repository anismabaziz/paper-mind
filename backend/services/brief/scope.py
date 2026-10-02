"""
The scope a brief reads: exactly two Documents, and nothing else.

A brief answers a question across Documents, so it has to be told which ones.
That is a narrowing, not a convenience: the model gets no way to widen it. The
two Documents are presented to it under short labels, and every search it asks
for names one of those labels. A label outside the pair is not an unknown
Document that would be looked up — it is refused, because looking it up is the
whole thing being prevented.

This module owns that pair and the refusals around it, so the check happens once
rather than in each tool. A Document that is being deleted, is being reindexed,
or whose index no longer matches the running configuration is refused here for
the same reasons chat refuses it: the vectors behind it are either going away or
no longer say what the app serves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.indexing.readiness import (
    BRIEF_REFUSAL_MESSAGES,
    refuse,
)
from services.indexing.state import record_stale_reason

#: A brief is a cross-Document question over a chosen pair. Not one, not three.
#: This is the default a caller falls back to; the brief's own limits carry the
#: number that was actually enforced, and the route passes them in.
BRIEF_DOCUMENT_COUNT = 2

#: What the model is shown instead of a storage filename. The filename is the
#: Document's identity in the store, and a model that could read it could ask
#: for a Document outside its scope by naming it.
SCOPE_LABELS = ("A", "B")


@dataclass(frozen=True)
class ScopedDocument:
    """One Document in a brief's scope, as both the reader and the model see it."""

    label: str
    filename: str
    document_id: str
    title: str
    index_generation: int | None

    def to_dict(self) -> dict[str, Any]:
        """Return the Document as the scope summary and the trace record it."""
        return {
            "label": self.label,
            "title": self.title,
            "document_id": self.document_id,
            "index_generation": self.index_generation,
        }


@dataclass(frozen=True)
class BriefScope:
    """
    The pair of Documents one brief may read, and the labels it reads them by.

    ``filenames`` is kept for the retrieval calls, which filter the store by the
    Document's own key. The model never sees it: ``describe`` is what the prompt
    is built from, and it names the two Documents by label and title only.
    """

    documents: tuple[ScopedDocument, ...]
    filenames: tuple[str, ...]

    @property
    def labels(self) -> tuple[str, ...]:
        """Return the labels the model may search under."""
        return tuple(document.label for document in self.documents)

    def document_for_label(self, label: str) -> ScopedDocument | None:
        """Return the Document a label names, or None when it names nothing here."""
        wanted = str(label).strip().upper()
        return next(
            (document for document in self.documents if document.label == wanted),
            None,
        )

    def describe(self) -> str:
        """Return the scope as the model is shown it, by label and title."""
        return "\n".join(
            f"[{document.label}] {document.title}" for document in self.documents
        )

    def to_dict(self) -> list[dict[str, Any]]:
        """Return the scope as the interface and the trace record it."""
        return [document.to_dict() for document in self.documents]


def resolve_scope(
    *,
    repositories: Any,
    settings: Any,
    filenames: list[str],
    limit: int = BRIEF_DOCUMENT_COUNT,
) -> tuple[BriefScope | None, dict[str, Any] | None]:
    """
    Return the brief's scope, or the refusal that stops the brief before it runs.

    Every refusal here is one the reader can act on, and none of them is a model
    failure: an unknown Document, one being deleted, one being reindexed, one
    whose index is stale, and one that was never indexed all mean the question
    cannot be asked yet, and each names which Document and what to do about it.

    The pair is checked before anything is retrieved, so a brief over one stale
    Document never spends a model call discovering it.
    """
    wanted = [str(name).strip() for name in filenames if str(name).strip()]
    if len(wanted) != limit or len(set(wanted)) != limit:
        return None, {
            "status": 400,
            "category": "brief_scope_invalid",
            "error": (
                f"A research brief compares exactly {limit} Documents. "
                f"Select {limit} different Documents."
            ),
        }

    scoped: list[ScopedDocument] = []
    for label, filename in zip(SCOPE_LABELS[:limit], wanted):
        file_record = repositories.files.get_file(filename)
        ingestion_job = repositories.ingestion_jobs.get_latest(filename)
        refusal = refuse(file_record, ingestion_job, settings)
        if file_record is not None:
            record_stale_reason(repositories.files, filename, file_record, settings)
        if refusal is not None:
            status, template = BRIEF_REFUSAL_MESSAGES[refusal.category]
            return None, {
                "status": status,
                "category": refusal.category,
                "error": template.format(label=label),
                "label": label,
                **refusal.detail,
            }
        file_record = file_record or {}
        scoped.append(
            ScopedDocument(
                label=label,
                filename=filename,
                document_id=file_record["id"],
                title=file_record.get("title") or file_record["filename"],
                index_generation=file_record.get("index_generation"),
            )
        )

    return (
        BriefScope(documents=tuple(scoped), filenames=tuple(wanted)),
        None,
    )
