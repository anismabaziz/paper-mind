"""
Title derivation for stored documents.

Priority:
1. PDF metadata ``Title`` via ``pymupdf`` (trimmed, non-empty).
2. Original filename without extension (trimmed, non-empty).
3. First non-empty heading-like line from the PDF's first pages.
4. ``"Untitled"`` as a last resort.
"""

import io
import logging
import os
import re
from typing import Protocol

MAX_TITLE_LEN = 200

_HEX_TITLE_RE = re.compile(r"^[0-9a-fA-F]{32}$")
log = logging.getLogger(__name__)


def is_hex_like_title(title: str | None) -> bool:
    """Return True when title looks like a storage hex name, not a human title."""
    if not title:
        return False
    stripped = title.strip()
    if _HEX_TITLE_RE.match(stripped):
        return True
    base, _ = os.path.splitext(stripped)
    return bool(_HEX_TITLE_RE.match(base))


def _normalize_pdf_bytes(pdf_bytes: bytes | None) -> bytes | None:
    """Return raw PDF bytes, handling bytearray and file-like wrappers."""
    if pdf_bytes is None:
        return None
    data = pdf_bytes
    if isinstance(data, bytearray):
        data = bytes(data)
    if hasattr(data, "read"):
        try:
            data = data.read()
        except Exception:
            return None
    if not data:
        return None
    return data


def _extract_pdf_metadata_title(pdf_bytes: bytes) -> str | None:
    """Return trimmed PDF metadata Title via pymupdf, or None."""
    try:
        import pymupdf

        with pymupdf.open("pdf", io.BytesIO(pdf_bytes)) as doc:
            title = (doc.metadata or {}).get("title") or ""
            title = title.strip()
            return title or None
    except Exception:
        return None


def _extract_first_heading(pdf_bytes: bytes) -> str | None:
    """Return first non-empty line from PDF text, truncated, or None."""
    try:
        import pymupdf

        with pymupdf.open("pdf", io.BytesIO(pdf_bytes)) as doc:
            for page in doc:
                text = page.get_text() or ""
                for line in text.splitlines():
                    heading = line.strip()
                    if heading:
                        if len(heading) > MAX_TITLE_LEN:
                            heading = heading[: MAX_TITLE_LEN - 3] + "..."
                        return heading
            return None
    except Exception:
        return None


def derive_title(pdf_bytes_or_none: bytes | None, original_filename: str | None) -> str:
    """Derive a human title from PDF bytes and the original filename."""
    pdf_bytes = _normalize_pdf_bytes(pdf_bytes_or_none)

    # 1. PDF metadata Title via pymupdf
    if pdf_bytes is not None:
        title = _extract_pdf_metadata_title(pdf_bytes)
        if title:
            return title

    # 2. Original filename without extension
    if original_filename:
        base = os.path.basename(original_filename.strip())
        name, _ = os.path.splitext(base)
        name = name.strip()
        if name:
            return name

    # 3. First heading heuristic
    if pdf_bytes is not None:
        heading = _extract_first_heading(pdf_bytes)
        if heading:
            return heading

    return "Untitled"


class TitleStorage(Protocol):
    """Read persisted bytes for a stored document."""

    def open(self, filename: str):
        """Return the stored document bytes."""
        ...


class TitleRepository(Protocol):
    """Persist a replacement title for a stored document."""

    def set_file_title(self, filename: str, title: str) -> None:
        """Replace the document title."""
        ...


def _needs_title_backfill(title: str | None) -> bool:
    return title is None or is_hex_like_title(title)


def backfill_title(
    storage: TitleStorage,
    repository: TitleRepository,
    filename: str,
    title: str | None,
    original_filename: str | None,
) -> str | None:
    """Derive and persist a title for legacy stored document metadata."""
    if not _needs_title_backfill(title):
        return title
    try:
        raw = storage.open(filename)
    except Exception as exc:
        log.warning("backfill read failed for %s: %s", filename, exc)
        return title
    try:
        derived = derive_title(raw, original_filename)
    except Exception as exc:
        log.warning("backfill derive failed for %s: %s", filename, exc)
        return title
    if derived and derived != title:
        try:
            repository.set_file_title(filename, derived)
        except Exception as exc:
            log.warning("backfill persist failed for %s: %s", filename, exc)
        return derived
    return title
