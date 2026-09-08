"""
Title derivation for stored documents.

Priority:
1. PDF metadata ``Title`` via ``pymupdf`` (trimmed, non-empty).
2. Original filename without extension (trimmed, non-empty).
3. First non-empty heading-like line from the PDF's first pages.
4. ``"Untitled"`` as a last resort.
"""

import io
import os

MAX_TITLE_LEN = 200


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


def derive_title(
    pdf_bytes_or_none: bytes | None, original_filename: str | None
) -> str:
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
