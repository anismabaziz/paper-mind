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


class UnknownDocumentFormat(ValueError):
    pass


class DocumentParser:
    @classmethod
    def for_filename(cls, filename: str):
        ext = os.path.splitext(filename)[1].lower()
        parser = _PARSERS.get(ext)
        if parser is None:
            supported = ", ".join(sorted(_PARSERS))
            raise UnknownDocumentFormat(
                f"No parser for {ext!r}; supported formats: {supported}"
            )
        return parser

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
        """
        parser = cls.for_filename(filename)
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
