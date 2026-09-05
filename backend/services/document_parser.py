"""
    Document parser seam: per-format extraction, shared chunking.

    Extraction lives behind :class:`DocumentParser` so a new format becomes a
    new parser rather than a rewrite. Chunking is intentionally owned by the
    seam itself, not by any parser, so swapping parsers cannot silently change
    chunk sizes or overlap.
"""

import hashlib
import os
from dataclasses import dataclass

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from services.pdf_service import PDFParser


@dataclass(frozen=True)
class Chunk:
    """Bundled chunk payload traveling together through ingestion."""

    text: str
    page_no: int | None
    chunk_index: int
    content_hash: str

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# cl100k_base is the tokenizer for gpt-4 / embeddings; stable, no download.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_ENCODING.encode(text))

_PARSERS = {".pdf": PDFParser}

# Re-export for callers that probe the registry (spec pins _PARSERS shape)
try:
    from services.docling_parser import DoclingParser as _DoclingParser

    _PARSERS[".pdf:docling"] = _DoclingParser  # optional layout-aware branch
except Exception:
    pass


def _should_use_docling(filename: str, file_bytes: bytes | None) -> bool:
    """Delegate to :mod:`services.pdf_heuristics` to keep this seam focused."""
    from services.pdf_heuristics import should_use_docling

    return should_use_docling(filename, file_bytes)


# Backward-compat re-exports so existing probes/tests keep working
try:
    from services.pdf_heuristics import (
        has_borderless_table as _has_borderless_table,
    )
    from services.pdf_heuristics import (
        is_image_only_pdf as _is_image_only_pdf,
    )
    from services.pdf_heuristics import (
        is_two_column_pdf as _is_two_column_pdf,
    )
except Exception:  # pragma: no cover
    pass


class UnknownDocumentFormat(ValueError):
    pass


class DocumentParser:
    @classmethod
    def for_filename(cls, filename: str, file_bytes: bytes | None = None):
        """
        Resolve a parser for ``filename``.

        When ``file_bytes`` is provided and the lightweight heuristic detects an
        image-only / borderless-table / 2-col PDF, returns :class:`DoclingParser`
        (if installed); otherwise returns :class:`PDFParser` (pymupdf fast path).
        The ``file_bytes`` argument is optional so existing callers
        (``for_filename("paper.pdf")``) keep working.
        """
        ext = os.path.splitext(filename)[1].lower()
        if ext != ".pdf":
            parser = _PARSERS.get(ext)
            if parser is None:
                supported = ", ".join(sorted(_PARSERS))
                raise UnknownDocumentFormat(
                    f"No parser for {ext!r}; supported formats: {supported}"
                )
            return parser

        # PDF: may route to Docling via heuristic
        if file_bytes is not None and _should_use_docling(filename, file_bytes):
            try:
                from services.docling_parser import DoclingParser

                return DoclingParser
            except Exception:
                pass
        return _PARSERS[".pdf"]

    @staticmethod
    def split_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=_token_len,
            is_separator_regex=False,
        )
        texts = text_splitter.create_documents([text])
        return [doc.page_content for doc in texts]

    @staticmethod
    def split_pages(page_texts: list[str], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        """
            Split per-page texts while preserving page numbers.

            Returns a list of (chunk_text, page_no) tuples. Page numbers are
            1-indexed. Each page is chunked independently so a chunk never
            straddles two pages and its page_no is unambiguous.
        """
        chunks_with_page: list[tuple[str, int]] = []
        for page_no, page_text in enumerate(page_texts, start=1):
            if not page_text or not page_text.strip():
                continue
            page_chunks = DocumentParser.split_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for chunk in page_chunks:
                chunks_with_page.append((chunk, page_no))
        return chunks_with_page

    @classmethod
    def get_chunk_objects(cls, filename: str, file_bytes: bytes) -> list[Chunk]:
        """Parse and chunk, returning bundled :class:`Chunk` objects."""
        chunks, page_numbers = cls.get_chunks(filename, file_bytes)
        return [
            Chunk(
                text=chunk,
                page_no=page_numbers[i],
                chunk_index=i,
                content_hash=hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
            )
            for i, chunk in enumerate(chunks)
        ]

    @classmethod
    def get_chunks(cls, filename: str, file_bytes: bytes):
        """
            Parse and chunk a file, returning (chunks, page_numbers).

            Page-aware when the parser exposes extract_pages; otherwise falls
            back to flat text. Single seam so app and evaluator share the
            same logic. Prefer :meth:`get_chunk_objects` for new code — the
            parallel lists are a data clump.

            Routing: when ``file_bytes`` looks image-only / 2-col / borderless-
            table and Docling is installed, the Docling branch is tried first
            (Markdown with hierarchy + tables as ``| col |``, header/footer
            dedup >70%, page_no preserved). On any Docling failure the pymupdf
            fast path is used so ingestion never breaks.
        """
        # Try heuristic Docling branch first when warranted, before the plain
        # extension lookup. This keeps two-column / table PDFs correct without
        # paying Docling cost for single-column born-digital PDFs.
        if file_bytes is not None and _should_use_docling(filename, file_bytes):
            try:
                from services.docling_parser import DoclingParser

                page_texts = DoclingParser.extract_pages(file_bytes)
                chunks_with_page = cls.split_pages(page_texts)
                chunks = [c for c, _ in chunks_with_page]
                page_numbers = [p for _, p in chunks_with_page]
                if chunks:
                    return chunks, page_numbers
            except Exception:
                pass

        parser = cls.for_filename(filename, file_bytes)
        if hasattr(parser, "extract_pages"):
            try:
                page_texts = parser.extract_pages(file_bytes)
                chunks_with_page = cls.split_pages(page_texts)
                chunks = [c for c, _ in chunks_with_page]
                page_numbers = [p for _, p in chunks_with_page]
                if chunks:
                    return chunks, page_numbers
            except Exception:
                pass
        text = parser.extract_text(file_bytes)
        chunks = cls.split_text(text)
        return chunks, [None] * len(chunks)
