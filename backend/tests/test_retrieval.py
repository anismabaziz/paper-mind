"""Tests for the document parser seam and retrieval shaping.

The parser seam owns extraction and chunking; the PDF implementation is one
parser among future ones. Retrieval shaping turns raw vector matches into a
deduped, score-ordered, bounded source list. Both run against fakes or
in-memory data only.
"""

import pytest

import config
from services import document_parser
from services.document_parser import DocumentParser
from services.vector_service import MAX_RETRIEVED_SOURCES, VectorService


class TestDocumentParserSeam:
    def test_pdf_filename_resolves_the_pdf_parser(self):
        assert DocumentParser.for_filename("paper.pdf") is not None

    def test_unknown_format_is_rejected_with_a_readable_error(self):
        with pytest.raises(document_parser.UnknownDocumentFormat) as exc:
            DocumentParser.for_filename("scan.docx")
        assert ".docx" in str(exc.value)

    def test_pdf_extraction_normalizes_whitespace_per_page(self):
        parser = DocumentParser.for_filename("paper.pdf")
        text = parser.extract_text(self._two_page_pdf())

        assert "first page words" in text
        assert "second page words" in text
        assert "\n" not in text, "page text must be normalized before joining"
        assert "  " not in text

    def test_split_text_defaults_preserve_chunking_behavior(self):
        text = " ".join(f"word{i}" for i in range(3000))

        chunks = DocumentParser.split_text(text)

        assert len(chunks) > 1
        assert all(len(chunk) <= 600 for chunk in chunks)
        # Defaults must be pinned: calling with no args equals an explicit
        # 600/100 call, so a refactor cannot silently change chunking.
        assert chunks == DocumentParser.split_text(
            text, chunk_size=600, chunk_overlap=100
        )

    @staticmethod
    def _two_page_pdf() -> bytes:
        import pymupdf

        doc = pymupdf.open()
        for words in ("first  page\nwords", "second\npage\n\nwords"):
            page = doc.new_page()
            page.insert_text((72, 72), words)
        return doc.tobytes()


class FakeVectorIndex:
    def __init__(self, matches):
        self._matches = matches
        self.queries = []

    def query(self, vector, top_k, include_metadata, filter):
        self.queries.append({"top_k": top_k, "filter": filter})
        return {"matches": self._matches}


class TestRetrievalShaping:
    @staticmethod
    def run_shaping(matches):
        index = FakeVectorIndex(matches)
        original = config.__dict__.get("_pinecone_index")
        config._pinecone_index = index
        try:
            return VectorService.query_vectors([0.1], "doc.pdf")
        finally:
            config._pinecone_index = original

    @staticmethod
    def match(content, score, chunk_index=0, document="doc.pdf"):
        return {
            "id": f"v-{content}-{score}",
            "score": score,
            "metadata": {
                "content": content,
                "pdf_name": document,
                "chunk_index": chunk_index,
            },
        }

    def test_results_are_ordered_by_score_descending(self):
        matches = [
            self.match("low", 0.10, 0),
            self.match("high", 0.90, 1),
            self.match("mid", 0.50, 2),
        ]

        sources = self.run_shaping(matches)

        assert [s["content"] for s in sources] == ["high", "mid", "low"]

    def test_duplicate_content_is_deduped_keeping_the_best_score(self):
        matches = [
            self.match("same text", 0.40, 0),
            self.match("same text", 0.80, 1),
            self.match("unique", 0.60, 2),
        ]

        sources = self.run_shaping(matches)

        assert [s["content"] for s in sources] == ["same text", "unique"]
        assert sources[0]["score"] == 0.80

    def test_results_are_bounded(self):
        matches = [self.match(f"chunk {i}", 1.0 - i / 10, i) for i in range(10)]

        sources = self.run_shaping(matches)

        assert len(sources) == MAX_RETRIEVED_SOURCES
        assert len(sources) < len(matches)

    def test_matches_without_content_are_dropped(self):
        matches = [
            {"id": "v-empty", "score": 0.9, "metadata": {"pdf_name": "doc.pdf"}},
            self.match("real", 0.5, 0),
        ]

        sources = self.run_shaping(matches)

        assert [s["content"] for s in sources] == ["real"]

    def test_query_is_filtered_to_the_document(self):
        index = FakeVectorIndex([])
        original = config.__dict__.get("_pinecone_index")
        config._pinecone_index = index
        try:
            VectorService.query_vectors([0.1], "doc.pdf")
        finally:
            config._pinecone_index = original

        assert index.queries[0]["filter"] == {"pdf_name": "doc.pdf"}
