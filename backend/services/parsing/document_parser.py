"""
Document parsing: the parser abstraction, routing, and shared chunking.

:class:`DocumentParser` is the abstraction every format parser implements:
one string per page, with row boundaries preserved so the chunker can keep
table rows together. :func:`resolve_parser` routes a filename (and, for
PDFs, lightweight layout heuristics) to a parser instance. Chunking is
intentionally owned by :class:`TokenChunker`, shared by every parser, so
swapping parsers cannot silently change chunk sizes or overlap.
:class:`DocumentIngestor` composes routing and chunking into the single
entry ingestion callers use.
"""

import hashlib
import io
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from services.parsing.pdf_heuristics import should_use_docling

log = logging.getLogger(__name__)

# cl100k_base is the tokenizer for gpt-4 / embeddings; stable, no download.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_ENCODING.encode(text))


def _pdf_page_count(file_bytes: bytes) -> int:
    """Count PDF pages; 0 when uncountable, which callers treat as unverified."""
    try:
        import pymupdf

        with pymupdf.open("pdf", io.BytesIO(file_bytes)) as doc:
            return len(doc)
    except Exception as exc:
        log.warning(
            "page count unavailable, treating provenance as unverified: %s", exc
        )
        return 0


@dataclass(frozen=True)
class ParseResult:
    """Chunks plus whether page provenance survived parsing."""

    chunks: list[str]
    page_numbers: list[int | None]
    degraded: bool
    reason: str | None = None


@dataclass(frozen=True)
class Chunk:
    """Bundled chunk payload traveling together through ingestion."""

    text: str
    page_no: int | None
    chunk_index: int
    content_hash: str


class UnknownDocumentFormat(ValueError):
    """Raised when no parser can handle a filename's extension."""


class DocumentParser(ABC):
    """One string per page, plus a flat-text view, for a document format."""

    @abstractmethod
    def extract_pages(self, file_bytes: bytes) -> list[str]:
        """
        Extract one string per page.

        Table rows are kept as separate lines (newlines preserved) so a
        chunker can keep row boundaries.
        """

    @abstractmethod
    def extract_text(self, file_bytes: bytes) -> str:
        """Flat text for backward compatibility (joined pages)."""


class TokenChunker:
    """Token-based chunking shared by every parser."""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        """Pin the chunk size and overlap for this chunker's lifetime."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text) -> list[str]:
        """Split one text into token-bounded, overlapping chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=_token_len,
            is_separator_regex=False,
        )
        texts = text_splitter.create_documents([text])
        return [doc.page_content for doc in texts]

    def split_pages(self, page_texts: list[str]) -> list[tuple[str, int]]:
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
            page_chunks = self.split_text(page_text)
            for chunk in page_chunks:
                chunks_with_page.append((chunk, page_no))
        return chunks_with_page


def resolve_parser(
    filename: str, file_bytes: bytes | None = None, use_docling: str = "auto"
) -> DocumentParser:
    """
    Resolve a parser instance for ``filename``.

    When ``file_bytes`` is provided and the lightweight heuristic detects an
    image-only / borderless-table / 2-col PDF, returns the Docling parser
    (if installed); otherwise returns the pymupdf fast path. The
    ``file_bytes`` argument is optional so ``resolve_parser("paper.pdf")``
    keeps working for pure lookups.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext != ".pdf":
        raise UnknownDocumentFormat(
            f"No parser for {ext!r}; supported formats: ['.pdf']"
        )

    # PDF: may route to Docling via heuristic
    if file_bytes is not None and should_use_docling(filename, file_bytes, use_docling):
        try:
            from services.parsing.docling_parser import DoclingParser

            return DoclingParser()
        except Exception as exc:
            log.warning("docling parser unavailable for %r: %s", filename, exc)
    from services.parsing.pdf_service import PDFParser

    return PDFParser()


class DocumentIngestor:
    """
    Parse-and-chunk entry point for ingestion.

    Routes a document to a parser (heuristics-driven for PDFs) and chunks it
    with the shared token chunker. Settings values (Docling routing, chunk
    size, overlap) arrive through the constructor.
    """

    def __init__(self, use_docling: str, chunk_size: int, chunk_overlap: int):
        """Bind the routing mode and the shared chunker's settings values."""
        self._use_docling = use_docling
        self._chunker = TokenChunker(chunk_size, chunk_overlap)

    def resolve(self, filename: str, file_bytes: bytes | None = None) -> DocumentParser:
        """Resolve the parser for one document."""
        return resolve_parser(filename, file_bytes, self._use_docling)

    def get_chunks(
        self, filename: str, file_bytes: bytes
    ) -> tuple[list[str], list[int | None]]:
        """
        Parse and chunk a file, returning (chunks, page_numbers).

        Page-aware when the parser succeeds; falls back to flat text when
        page extraction fails. Routing: when ``file_bytes`` looks image-only
        / 2-col / borderless-table and Docling is installed, the Docling
        branch is tried first (Markdown with hierarchy + tables as
        ``| col |``, header/footer dedup >70%, page_no preserved). On any
        Docling failure the pymupdf fast path is used so ingestion never
        breaks. Prefer :meth:`get_chunk_objects` for new code — the parallel
        lists are a data clump.
        """
        result = self.get_chunks_with_status(filename, file_bytes)
        return result.chunks, result.page_numbers

    def get_chunks_with_status(self, filename: str, file_bytes: bytes) -> ParseResult:
        """
        Parse and chunk, reporting whether page provenance was degraded.

        ``degraded`` is True when page numbers fell back to null so callers
        can surface a warning; ``reason`` names the failed branch.
        """
        # Try heuristic Docling branch first when warranted, before the plain
        # extension lookup. This keeps two-column / table PDFs correct without
        # paying Docling cost for single-column born-digital PDFs.
        if should_use_docling(filename, file_bytes, self._use_docling):
            try:
                from services.parsing.docling_parser import DoclingParser

                page_texts = DoclingParser().extract_pages(file_bytes)
                chunks_with_page = self._chunker.split_pages(page_texts)
                chunks = [c for c, _ in chunks_with_page]
                page_numbers = [p for _, p in chunks_with_page]
                if chunks:
                    if len(page_texts) == 1 and _pdf_page_count(file_bytes) != 1:
                        # Provenance collapsed: one chunk-page for a multi-page
                        # (or uncountable) document. Page 1 would be false
                        # provenance, so propagate null pages as degraded.
                        log.warning(
                            "docling collapsed %r to a single page without "
                            "provenance; chunks carry null page numbers",
                            filename,
                        )
                        return ParseResult(
                            chunks=chunks,
                            page_numbers=[None] * len(chunks),
                            degraded=True,
                            reason="docling-provenance-unavailable",
                        )
                    return ParseResult(
                        chunks=chunks, page_numbers=page_numbers, degraded=False
                    )
            except Exception as exc:
                log.warning(
                    "docling branch failed for %r, trying fast path: %s", filename, exc
                )

        parser = self.resolve(filename, file_bytes)
        try:
            page_texts = parser.extract_pages(file_bytes)
            chunks_with_page = self._chunker.split_pages(page_texts)
            chunks = [c for c, _ in chunks_with_page]
            page_numbers = [p for _, p in chunks_with_page]
            if chunks:
                return ParseResult(
                    chunks=chunks, page_numbers=page_numbers, degraded=False
                )
        except Exception as exc:
            log.warning(
                "page-aware parse failed for %r, falling back to flat text "
                "with null pages: %s",
                filename,
                exc,
            )
        text = parser.extract_text(file_bytes)
        chunks = self._chunker.split_text(text)
        if chunks:
            log.warning(
                "degraded parse for %r: %d chunks carry null page numbers",
                filename,
                len(chunks),
            )
        return ParseResult(
            chunks=chunks,
            page_numbers=[None] * len(chunks),
            degraded=True,
            reason="page-extract-failed",
        )

    def get_chunk_objects(self, filename: str, file_bytes: bytes) -> list[Chunk]:
        """Parse and chunk, returning bundled :class:`Chunk` objects."""
        chunks, page_numbers = self.get_chunks(filename, file_bytes)
        return [
            Chunk(
                text=chunk,
                page_no=page_numbers[i],
                chunk_index=i,
                content_hash=hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
            )
            for i, chunk in enumerate(chunks)
        ]

    def split_text(self, text) -> list[str]:
        """Split one text with the configured chunk size and overlap."""
        return self._chunker.split_text(text)

    def split_pages(self, page_texts: list[str]) -> list[tuple[str, int]]:
        """Split per-page texts, preserving page numbers."""
        return self._chunker.split_pages(page_texts)
