"""
What one brief has read, and the ids a later part of it cites.

Evidence arrives in pieces: a search returns five Passages from one Document,
the next search returns five from the other, and the model may ask for the same
Passage twice. The ids handed out here are the same for the whole run, so a
citation written in the first minute and one written in the last name the same
Passage — and a Passage found twice is one piece of evidence, not two.

The id is assigned when the Passage is first admitted, not when it is first
ranked, because what makes a citation checkable is that the id a claim names is
the id under which the model was shown that text. An id allocated from the
search position would be renumbered by the next search and every earlier claim
would silently point somewhere else.

Ordering is the order of admission, and the rank stored alongside is the rank
within the search that found it. Both are recorded: the rank is where the
retrieval put this Passage for that query, and presenting it as the evidence's
own standing would be claiming a calibration the retrieval never measured.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable

#: The prefix every evidence id in a brief carries. Distinct from a chat Turn's
#: ``S1`` because the two are collected differently: ``S`` ids come from one
#: retrieval of one Document, these accumulate across a whole brief.
EVIDENCE_PREFIX = "E"

#: How much of a Passage a search shows. Enough for the model to judge whether a
#: Passage is worth reading, and bounded so a search's cost does not scale with
#: how long the Passages it matched happen to be. The rest is what
#: ``read_passages`` is for.
EXCERPT_CHARS = 600


@dataclass
class Evidence:
    """
    One Passage a brief is holding, with the ids a claim needs to reach it.

    ``content`` is what the model has actually been shown. A search returns a
    Passage in part, so what is stored at first is that part; reading the
    evidence replaces it with the whole Passage and raises ``read``. The reader
    is told which is which, because "read" and "found" are not the same claim.
    """

    evidence_id: str
    label: str
    document_id: str
    title: str
    #: The Document's storage filename, which is its identity in the store and
    #: is never what a reader is shown.
    document: str
    chunk_index: int
    page: int | None
    #: Where this Passage landed in the search that found it, counting from 1.
    #: It is a rank within that query, not a score of being right.
    rank: int
    #: The order the brief admitted it in, which is the order it is shown in.
    position: int
    method: str
    #: The Passage as the store holds it, in full.
    content: str
    read: bool = False

    @property
    def excerpt(self) -> str:
        """Return the part of the Passage a search showed the model."""
        return self.content[:EXCERPT_CHARS]

    @property
    def shown(self) -> str:
        """Return what the model has actually been shown, in full or in part."""
        return self.content if self.read else self.excerpt

    def to_dict(self) -> dict[str, Any]:
        """Return the evidence as the terminal event and the trace record it."""
        return {
            "evidence_id": self.evidence_id,
            "label": self.label,
            "document_id": self.document_id,
            "document": self.document,
            "title": self.title,
            "chunk_index": self.chunk_index,
            "page": self.page,
            "rank": self.rank,
            "position": self.position,
            "method": self.method,
            "content": self.content,
            "read": self.read,
        }


def _content_hash(content: str) -> str:
    """Return the digest the store would have recorded for this Passage."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _page_of(source: dict[str, Any]) -> int | None:
    """Return the Page a Passage came from, tolerating either key the store uses."""
    page = source["page"] if "page" in source else source.get("page_no")
    return None if page is None else page


@dataclass
class EvidenceLedger:
    """
    The evidence one brief has collected, and the ids naming it.

    Dedupe is by content hash, which is what survives two searches returning
    overlapping windows: the same Passage is admitted once and keeps the id and
    the rank it was first given. A Passage the store gave no hash for is hashed
    here instead, so a store that cannot hash its payloads costs a little work
    rather than renumbering an id a claim already names.
    """

    _evidence: dict[str, Evidence] = field(default_factory=dict)
    _by_hash: dict[tuple[str, str], str] = field(default_factory=dict)

    def admit(
        self,
        sources: Iterable[dict[str, Any]],
        *,
        document: Any,
        method: str,
    ) -> list[Evidence]:
        """
        Admit what one search returned, keeping ids stable across the run.

        ``document`` is the :class:`~services.brief.scope.ScopedDocument` the
        search was scoped to, which is how a Passage is attributed to A or B
        without the model having said which one it meant.
        """
        admitted: list[Evidence] = []
        for rank, source in enumerate(sources, start=1):
            content = source.get("content") or ""
            if not content.strip():
                continue
            content_hash = str(source.get("content_hash") or _content_hash(content))
            key = (document.document_id, content_hash)
            existing = self._by_hash.get(key)
            if existing is not None:
                admitted.append(self._evidence[existing])
                continue
            evidence = Evidence(
                evidence_id=f"{EVIDENCE_PREFIX}{len(self._evidence) + 1}",
                label=document.label,
                document_id=document.document_id,
                title=document.title,
                document=document.filename,
                chunk_index=int(source.get("chunk_index") or 0),
                page=_page_of(source),
                rank=rank,
                position=len(self._evidence) + 1,
                method=method,
                content=content,
            )
            self._evidence[evidence.evidence_id] = evidence
            self._by_hash[key] = evidence.evidence_id
            admitted.append(evidence)
        return admitted

    def get(self, evidence_id: str) -> Evidence | None:
        """Return the evidence an id names, or None when it names nothing held."""
        return self._evidence.get(str(evidence_id).strip().upper())

    def ids(self) -> tuple[str, ...]:
        """Return every evidence id this brief holds, in admission order."""
        return tuple(self._evidence)

    def read(self, evidence_id: str) -> Evidence | None:
        """
        Return one evidence with its whole Passage, marking it read.

        Reading is what turns a fragment into something a claim can rest on, so
        the flag is what tells a reader which of the evidence on screen the model
        actually had in full.
        """
        item = self._evidence.get(str(evidence_id).strip().upper())
        if item is None:
            return None
        item.read = True
        return item

    def held(self) -> tuple[Evidence, ...]:
        """Return every Passage the brief is holding, in admission order."""
        return tuple(self._evidence.values())

    def to_dict(self) -> list[dict[str, Any]]:
        """Return the held evidence as the terminal event records it."""
        return [item.to_dict() for item in self.held()]
