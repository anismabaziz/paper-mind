"""
Docling layout-aware parser (opt-in).

Keeps tables as Markdown tables and preserves two-column reading order without
requiring a paid OCR API. Heavy deps (``docling`` + ``granite-docling-258M``
~1.1GB) are optional — install with ``pip install ".[docling]"`` or
``uv sync --extra docling``. The model weights are cached via ``HF_HOME``
(Docker volume ``hf_cache`` in compose.yaml) so the download only happens
once.

The parser is reached through :class:`services.document_parser.DocumentParser`
(same interface as :class:`services.pdf_service.PDFParser`) so swapping parsers
cannot silently change chunk sizes. Tests mock this parser and never download.
"""

from __future__ import annotations

import importlib.util
import io
import re
from collections import Counter
from typing import List

# ---------------------------------------------------------------------------
# Header/footer dedup helper (pure python, no heavy deps)
# ---------------------------------------------------------------------------


def _strip_repeating_headers_footers(
    pages: List[str], threshold: float = 0.7
) -> List[str]:
    """
    Strip repeating headers/footers via dedup (>70% same position across pages).

    A line is considered a repeating header/footer if the exact same string
    appears at the same position (first line = header, last line = footer) on
    more than ``threshold`` fraction of pages. Handles 1-2 lines at top/bottom.
    Comparison is stripped but preserves original line for removal decision.

    Example: if "Confidential - Do Not Distribute" appears as the first line on
    8/10 pages (80% > 70%), it is removed from those pages.
    """
    if len(pages) <= 1:
        return pages

    split: List[List[str]] = [p.split("\n") for p in pages]
    n = len(pages)
    header_footer_offsets = [0, 1, -1, -2]
    to_strip: dict[int, set[str]] = {}

    for offset in header_footer_offsets:
        lines_at_offset: List[str] = []
        for lines in split:
            if not lines:
                lines_at_offset.append("")
                continue
            if offset >= 0:
                if offset < len(lines):
                    lines_at_offset.append(lines[offset].strip())
                else:
                    lines_at_offset.append("")
            else:
                idx = len(lines) + offset
                if idx >= 0:
                    lines_at_offset.append(lines[idx].strip())
                else:
                    lines_at_offset.append("")
        counter = Counter(l for l in lines_at_offset if l)
        for text, cnt in counter.items():
            if cnt / n > threshold:
                to_strip.setdefault(offset, set()).add(text)

    if not to_strip:
        return pages

    result: List[str] = []
    for lines in split:
        # Resolve offsets to absolute indices before mutating, so -1/-2
        # don't shift when one is popped before the other.
        indices_to_remove: set[int] = set()
        for offset, texts in to_strip.items():
            if offset >= 0:
                if offset < len(lines) and lines[offset].strip() in texts:
                    indices_to_remove.add(offset)
            else:
                idx = len(lines) + offset
                if 0 <= idx < len(lines) and lines[idx].strip() in texts:
                    indices_to_remove.add(idx)
        if not indices_to_remove:
            result.append("\n".join(lines))
            continue
        mutable = [l for i, l in enumerate(lines) if i not in indices_to_remove]
        result.append("\n".join(mutable))

    return result


def _try_import_docling() -> None:
    """Raise a helpful ImportError if docling is not installed."""
    if importlib.util.find_spec("docling") is None:
        raise ImportError(
            "Docling is not installed. Install the optional extra with "
            '`uv sync --extra docling` or `pip install ".[docling]"` '
            "(pulls `docling` + `granite-docling-258M` ~1.1GB, cached to "
            "HF_HOME / Docker volume `hf_cache`). "
            "Tests mock this parser so `uv run pytest` never downloads."
        )


def _is_docling_available() -> bool:
    return importlib.util.find_spec("docling") is not None


def _ensure_docling_available() -> None:
    _try_import_docling()


class DoclingParser:
    r"""
    Layout-aware PDF parser via Docling (MIT).

    - Outputs Markdown with hierarchy (headings) and tables as Markdown tables
      (``| col |``), keeping row boundaries as ``\\n``.
    - Strips repeating headers/footers via dedup (>70% same position).
    - Preserves ``page_no``: :meth:`extract_pages` returns one string per
      page (1-indexed page_no travels via ``DocumentParser``).
    - Heavy import is lazy so collection never triggers a download.
    """

    @staticmethod
    def is_available() -> bool:
        """Do is available."""
        return _is_docling_available()

    @staticmethod
    def extract_pages(pdf_content: bytes) -> List[str]:
        """
        Extract one Markdown string per page.

        Table rows are Markdown table rows (``| a | b |``), hierarchy is
        ``#``/``##`` headings, and repeating headers/footers are stripped.
        ``page_no`` is implicit via list index (1-indexed in the caller).
        """
        _ensure_docling_available()

        # Lazy imports — only when actually parsing
        from docling.document_converter import DocumentConverter

        # Docling expects a file-like or path; use BytesIO
        pdf_stream = io.BytesIO(pdf_content)

        converter = DocumentConverter()
        result = converter.convert(source=pdf_stream)
        doc = result.document

        # Try per-page markdown first (docling may expose pages with export)
        # Fallback 1: try doc.export_to_markdown with page-wise splitting
        # Fallback 2: full markdown split across page count
        pages_md: List[str] = []

        # Primary strategy: group elements by provenance page_no (preserves
        # layout and keeps tables intact). Falls back to a single-page
        # markdown export so tables are never cut mid-block.
        try:
            # Collect elements grouped by page_no via provenance
            # doc.texts / doc.tables often have `prov` with page_no
            from collections import defaultdict

            grouped: dict[int, List[str]] = defaultdict(list)
            has_prov = False

            # Texts
            for elem in getattr(doc, "texts", []):
                prov = getattr(elem, "prov", None)
                if prov:
                    # prov is list of ProvenanceItem with page_no
                    try:
                        page_no = (
                            prov[0].page_no if isinstance(prov, list) else prov.page_no
                        )
                        has_prov = True
                    except Exception:
                        page_no = 1
                else:
                    page_no = 1
                # Get markdown/text
                txt = getattr(elem, "text", None) or str(elem)
                label = getattr(elem, "label", None)
                if label and "heading" in str(label).lower():
                    level = getattr(elem, "level", None)
                    if level is None:
                        # Fallback: infer from label
                        lbl = str(label).lower()
                        if "title" in lbl:
                            level = 1
                        elif "h1" in lbl or "heading_1" in lbl:
                            level = 1
                        elif "h3" in lbl or "heading_3" in lbl:
                            level = 3
                        else:
                            level = 2
                    try:
                        lvl = max(1, min(6, int(level)))
                    except Exception:
                        lvl = 2
                    txt = f"{'#' * lvl} {txt}" if txt else txt
                grouped[page_no].append(txt)

            # Tables -> Markdown tables
            for tbl in getattr(doc, "tables", []):
                prov = getattr(tbl, "prov", None)
                if prov:
                    try:
                        page_no = (
                            prov[0].page_no if isinstance(prov, list) else prov.page_no
                        )
                        has_prov = True
                    except Exception:
                        page_no = 1
                else:
                    page_no = 1
                try:
                    md_table = (
                        tbl.export_to_markdown(doc)
                        if hasattr(tbl, "export_to_markdown")
                        else str(tbl)
                    )
                except Exception:
                    # Fallback: try doc export of table
                    try:
                        md_table = doc.export_to_markdown()
                    except Exception:
                        md_table = str(tbl)
                grouped[page_no].append(md_table)

            if has_prov and grouped:
                max_page = max(grouped)
                pages_md = []
                for pno in range(1, max_page + 1):
                    content = "\n\n".join(grouped.get(pno, []))
                    pages_md.append(content.strip())
                # If grouping produced empty pages, fall back
                if any(p.strip() for p in pages_md):
                    pages_md = _strip_repeating_headers_footers(pages_md)
                    return pages_md
        except Exception:
            pass

        # Fallback: single-page markdown (preserves tables and hierarchy).
        # We avoid proportional splitting across page count because it can
        # cut markdown tables mid-block and assigns arbitrary page_no. When
        # provenance is unavailable the content is correct but page_no is 1,
        # which still flows through chunk metadata.
        try:
            full_md = doc.export_to_markdown()
        except Exception:
            full_md = ""

        if not full_md or not full_md.strip():
            try:
                import pymupdf

                with pymupdf.open("pdf", io.BytesIO(pdf_content)) as pdf_doc:
                    pages_md = [""] * len(pdf_doc)
            except Exception:
                pages_md = [full_md.strip()]
            return pages_md

        # If docling embeds explicit page breaks, split there; otherwise keep
        # as a single page to avoid breaking tables.
        if "<!-- image -->" in full_md or "\n---\n" in full_md:
            try:
                import pymupdf

                with pymupdf.open("pdf", io.BytesIO(pdf_content)) as pdf_doc:
                    n_pages = len(pdf_doc)
            except Exception:
                n_pages = 1
            parts = re.split(r"\n---\n", full_md)
            if len(parts) == n_pages and n_pages > 1:
                pages_md = [p.strip() for p in parts]
                pages_md = _strip_repeating_headers_footers(pages_md)
                return pages_md

        pages_md = [full_md.strip()]
        pages_md = _strip_repeating_headers_footers(pages_md)
        return pages_md

    @staticmethod
    def extract_text(pdf_content: bytes) -> str:
        """
        Flat Markdown for backward compatibility (joined pages).

        Preserves Markdown tables and hierarchy; callers needing page_no should
        use :meth:`extract_pages`.
        """
        pages = DoclingParser.extract_pages(pdf_content)
        # Join with double newline to keep markdown table block boundaries
        flat = "\n\n".join(p.strip() for p in pages if p.strip())
        # Normalize: collapse 3+ newlines to 2, strip
        flat = re.sub(r"\n{3,}", "\n\n", flat)
        return flat.strip()

    # Alias for internal use
    @classmethod
    def _strip_headers_footers(cls, pages: List[str]) -> List[str]:
        return _strip_repeating_headers_footers(pages)
