"""
Lightweight PDF heuristics for layout-aware routing.

All checks use ``pymupdf`` only and stay under 50ms for a 30-page PDF,
versus seconds for a Docling layout pass. The heuristics decide whether
a PDF is image-only / two-column / borderless-table and therefore benefits
from :class:`services.docling_parser.DoclingParser`; others stay on the
:class:`services.pdf_service.PDFParser` fast path.
"""

from __future__ import annotations

import io
import os
import re


def is_image_only_pdf(file_bytes: bytes, threshold: float = 0.5) -> bool:
    """Return True if > threshold fraction of pages are image-only."""
    try:
        import pymupdf

        with pymupdf.open("pdf", io.BytesIO(file_bytes)) as doc:
            if len(doc) == 0:
                return False
            image_heavy = 0
            for page in doc:
                text = page.get_text().strip()
                has_images = len(page.get_images(full=True)) > 0
                if has_images and len(text) < 80:
                    image_heavy += 1
                elif not text and has_images:
                    image_heavy += 1
            return (image_heavy / len(doc)) >= threshold
    except Exception:
        return False


def is_two_column_pdf(file_bytes: bytes) -> bool:
    """Return True if any page has a two-column layout."""
    try:
        import pymupdf

        with pymupdf.open("pdf", io.BytesIO(file_bytes)) as doc:
            for page in doc:
                blocks = page.get_text("blocks")
                text_blocks = [b for b in blocks if b[4].strip()]
                if len(text_blocks) < 4:
                    continue
                width = page.rect.width
                mid = width / 2
                left = [b for b in text_blocks if b[2] < mid - 10]
                right = [b for b in text_blocks if b[0] > mid + 10]
                if len(left) >= 2 and len(right) >= 2:
                    for lb in left:
                        for rb in right:
                            y_overlap = not (lb[3] < rb[1] or rb[3] < lb[1])
                            if y_overlap:
                                return True
    except Exception:
        return False
    return False


def has_borderless_table(file_bytes: bytes) -> bool:
    """Heuristic for borderless tables: detected tables or multi-gap lines."""
    try:
        import pymupdf

        with pymupdf.open("pdf", io.BytesIO(file_bytes)) as doc:
            for page in doc:
                try:
                    tables = page.find_tables()
                    if tables and len(tables.tables) > 0:
                        return True
                except Exception:
                    pass
                text = page.get_text()
                lines = text.split("\n")
                table_like = 0
                for line in lines:
                    parts = re.split(r"\s{2,}", line.strip())
                    if len(parts) >= 3 and all(p.strip() for p in parts):
                        table_like += 1
                    elif (
                        "\t" in line
                        and len([p for p in line.split("\t") if p.strip()]) >= 3
                    ):
                        table_like += 1
                if table_like >= 3:
                    return True
    except Exception:
        return False
    return False


def should_use_docling(filename: str, file_bytes: bytes | None) -> bool:
    """
    Decide whether a PDF should go through Docling.

    - ``USE_DOCLING`` env: ``true`` forces Docling (if installed), ``false``
      disables it, ``auto`` (default) uses the lightweight heuristic.
    - When docling is not installed, always return False (graceful fallback).
    """
    use = os.getenv("USE_DOCLING", "auto").lower()
    if use in ("0", "false", "no", "off", "disable", "disabled"):
        return False
    try:
        from services.docling_parser import DoclingParser

        if not DoclingParser.is_available():
            return False
    except Exception:
        return False

    if use in ("1", "true", "yes", "on", "enable", "enabled", "force"):
        return True

    if file_bytes is None:
        return False
    if is_image_only_pdf(file_bytes):
        return True
    if is_two_column_pdf(file_bytes):
        return True
    if has_borderless_table(file_bytes):
        return True
    return False
