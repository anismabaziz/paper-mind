"""Document parser seam: per-format extraction, shared chunking.

Extraction lives behind :class:`DocumentParser` so a new format becomes a
new parser rather than a rewrite. Chunking is intentionally owned by the
seam itself, not by any parser, so swapping parsers cannot silently change
chunk sizes or overlap.
"""

import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

from services.pdf_service import PDFParser

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

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
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.create_documents([text])
        return [doc.page_content for doc in texts]
