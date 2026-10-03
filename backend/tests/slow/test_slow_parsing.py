"""Slow tests for real PDF parsing and chunking.

Uses the local pymupdf path only. The docling branch is skipped when the
optional dependency is missing. Needs ``pytest tests/slow``.
"""

import pytest

from conftest import make_pdf_bytes

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def ingestor():
    from services.parsing.document_parser import DocumentIngestor

    return DocumentIngestor("false", 512, 50)


class TestPymupdfParsing:
    def test_page_count(self, ingestor):
        pdf = make_pdf_bytes(["page one text", "page two text", "page three text"])
        assert ingestor.get_page_count("doc.pdf", pdf) == 3

    def test_chunks_carry_page_numbers(self, ingestor):
        pdf = make_pdf_bytes(["alpha beta gamma", "delta epsilon zeta"])
        chunks = ingestor.get_chunk_objects("doc.pdf", pdf)
        assert len(chunks) >= 2
        assert {chunk.page_no for chunk in chunks} == {1, 2}
        assert all(chunk.content_hash and chunk.text.strip() for chunk in chunks)

    def test_chunk_indices_ordered(self, ingestor):
        pdf = make_pdf_bytes(["some text here"] * 3)
        chunks = ingestor.get_chunk_objects("doc.pdf", pdf)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_long_document_splits_into_many_chunks(self, ingestor):
        from services.parsing.document_parser import TokenChunker

        chunker = TokenChunker(512, 50)
        texts = chunker.split_text("word " * 5000)
        assert len(texts) > 5
        assert all(text.strip() for text in texts)

    def test_unknown_extension_rejected(self, ingestor):
        from services.parsing.document_parser import UnknownDocumentFormat

        with pytest.raises(UnknownDocumentFormat):
            ingestor.get_chunk_objects("doc.txt", b"whatever")

    def test_status_reports_degraded_flag(self, ingestor):
        pdf = make_pdf_bytes(["readable text"])
        result = ingestor.get_chunks_with_status("doc.pdf", pdf)
        assert result.chunks
        assert result.degraded is False


class TestDoclingBranch:
    def test_skipped_without_extra(self):
        pytest.importorskip("docling")
