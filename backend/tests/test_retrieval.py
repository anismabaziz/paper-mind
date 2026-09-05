"""
Tests for the document parser and retrieval shaping.

The parser owns extraction and chunking; the PDF implementation is one
parser among future ones. Retrieval shaping turns raw vector matches into a
deduped, score-ordered, bounded source list. Both run against fakes or
in-memory data only.
"""

import pytest

import config
from services import document_parser
from services.document_parser import DocumentParser
from services.vector_service import MAX_RETRIEVED_SOURCES, VectorService


class TestDocumentParser:
    """TestDocumentParser."""

    def test_pdf_filename_resolves_the_pdf_parser(self):
        """Do test pdf filename resolves the pdf parser."""
        assert DocumentParser.for_filename("paper.pdf") is not None

    def test_unknown_format_is_rejected_with_a_readable_error(self):
        """Do test unknown format is rejected with a readable error."""
        with pytest.raises(document_parser.UnknownDocumentFormat) as exc:
            DocumentParser.for_filename("scan.docx")
        assert ".docx" in str(exc.value)

    def test_pdf_extraction_normalizes_whitespace_per_page(self):
        """Do test pdf extraction normalizes whitespace per page."""
        parser = DocumentParser.for_filename("paper.pdf")
        text = parser.extract_text(self._two_page_pdf())

        assert "first page words" in text
        assert "second page words" in text
        assert "\n" not in text, "page text must be normalized before joining"
        assert "  " not in text

    def test_split_text_defaults_preserve_chunking_behavior(self):
        """Do test split text defaults preserve chunking behavior."""
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        text = " ".join(f"word{i}" for i in range(3000))

        chunks = DocumentParser.split_text(text)

        assert len(chunks) > 1
        # Token-bounded: each chunk is ~512 tokens; last may be smaller.
        for chunk in chunks[:-1]:
            n = len(enc.encode(chunk))
            assert 487 <= n <= 537, f"chunk {n} tokens outside 512±25"
        assert len(enc.encode(chunks[-1])) <= 537
        # Defaults must be pinned: no-arg equals explicit 512/50.
        assert chunks == DocumentParser.split_text(
            text, chunk_size=512, chunk_overlap=50
        )
        # Overlap preservation: ~50 tokens across boundary (token-level check).
        t0 = enc.encode(chunks[0])
        t1 = enc.encode(chunks[1])
        tail_tokens = set(t0[-50:])
        head_tokens = set(t1[:60])
        assert len(tail_tokens & head_tokens) >= 15, (
            "overlap should preserve ~50 tokens across boundary"
        )

    def test_split_pages_preserves_page_numbers(self):
        """Do test split pages preserves page numbers."""
        import pymupdf

        doc = pymupdf.open()
        for i, words in enumerate(
            ["page one content " * 200, "page two content " * 200]
        ):
            page = doc.new_page()
            page.insert_text((72, 72), words)
        pdf_bytes = doc.tobytes()

        parser = DocumentParser.for_filename("paper.pdf")
        pages = parser.extract_pages(pdf_bytes)
        assert len(pages) == 2
        chunks_with_page = DocumentParser.split_pages(pages)
        # Every chunk knows its page
        assert all(page_no in (1, 2) for _, page_no in chunks_with_page)
        # Chunks from page 1 and page 2 both exist
        page_nos = {p for _, p in chunks_with_page}
        assert page_nos == {1, 2}

    def test_extract_pages_keeps_table_rows(self):
        """Do test extract pages keeps table rows."""
        import pymupdf

        doc = pymupdf.open()
        page = doc.new_page()
        # Simulate a table as lines
        page.insert_text((72, 72), "colA colB\nrow1 val1\nrow2 val2")
        pdf_bytes = doc.tobytes()

        parser = DocumentParser.for_filename("paper.pdf")
        pages = parser.extract_pages(pdf_bytes)
        assert len(pages) == 1
        # Rows preserved as newlines, not collapsed to spaces
        assert "colA" in pages[0]
        assert "\n" in pages[0]
        # Flat text still normalizes
        flat = parser.extract_text(pdf_bytes)
        assert "\n" not in flat

    @staticmethod
    def _two_page_pdf() -> bytes:
        import pymupdf

        doc = pymupdf.open()
        for words in ("first  page\nwords", "second\npage\n\nwords"):
            page = doc.new_page()
            page.insert_text((72, 72), words)
        return doc.tobytes()


class FakeVectorIndex:
    """FakeVectorIndex."""

    def __init__(self, matches):
        """Initialize."""
        self._matches = matches
        self.queries = []

    def query(self, vector, top_k, include_metadata, filter):
        """Do query."""
        self.queries.append({"top_k": top_k, "filter": filter})
        return {"matches": self._matches}


@pytest.fixture
def fake_index(monkeypatch):
    """Install a fake vector index behind config.vector_index."""
    installed = {}

    def install(matches):
        """Do install."""
        index = FakeVectorIndex(matches)
        # Patch the memo slot, not `vector_index` itself: setattr would read
        # the current value first, which triggers the lazy Pinecone builder.
        monkeypatch.setattr(config, "_pinecone_index", index)
        installed["index"] = index
        return index

    yield install


class TestRetrievalShaping:
    """TestRetrievalShaping."""

    def run_shaping(self, fake_index, matches):
        """Do run shaping."""
        index = fake_index(matches)
        return VectorService.query_vectors([0.1], "doc.pdf"), index

    @staticmethod
    def match(content, score, chunk_index=0, document="doc.pdf"):
        """Do match."""
        return {
            "id": f"v-{content}-{score}",
            "score": score,
            "metadata": {
                "content": content,
                "pdf_name": document,
                "chunk_index": chunk_index,
            },
        }

    def test_results_are_ordered_by_score_descending(self, fake_index):
        """Do test results are ordered by score descending."""
        matches = [
            self.match("low", 0.10, 0),
            self.match("high", 0.90, 1),
            self.match("mid", 0.50, 2),
        ]

        sources, _ = self.run_shaping(fake_index, matches)

        assert [s["content"] for s in sources] == ["high", "mid", "low"]

    def test_duplicate_content_is_deduped_keeping_the_best_score(self, fake_index):
        """Do test duplicate content is deduped keeping the best score."""
        matches = [
            self.match("same text", 0.40, 0),
            self.match("same text", 0.80, 1),
            self.match("unique", 0.60, 2),
        ]

        sources, _ = self.run_shaping(fake_index, matches)

        assert [s["content"] for s in sources] == ["same text", "unique"]
        assert sources[0]["score"] == 0.80

    def test_results_are_bounded(self, fake_index):
        """Do test results are bounded."""
        matches = [self.match(f"chunk {i}", 1.0 - i / 10, i) for i in range(10)]

        sources, _ = self.run_shaping(fake_index, matches)

        assert len(sources) == MAX_RETRIEVED_SOURCES
        assert len(sources) < len(matches)

    def test_matches_without_content_are_dropped(self, fake_index):
        """Do test matches without content are dropped."""
        matches = [
            {"id": "v-empty", "score": 0.9, "metadata": {"pdf_name": "doc.pdf"}},
            self.match("real", 0.5, 0),
        ]

        sources, _ = self.run_shaping(fake_index, matches)

        assert [s["content"] for s in sources] == ["real"]

    def test_query_is_filtered_to_the_document(self, fake_index):
        """Do test query is filtered to the document."""
        index = fake_index([])

        VectorService.query_vectors([0.1], "doc.pdf")

        assert index.queries[0]["filter"] == {"pdf_name": "doc.pdf"}


class TestChunkMetadata:
    """TestChunkMetadata."""

    def test_upsert_includes_page_no_and_content_hash(self, monkeypatch):
        """Do test upsert includes page no and content hash."""
        import hashlib

        import config

        captured = {}

        def fake_upsert(self, vectors):
            """Do fake upsert."""
            captured["vectors"] = vectors
            return {"upserted": len(vectors)}

        fake_index = type("Idx", (), {"upsert": fake_upsert})()
        monkeypatch.setattr(config, "_pinecone_index", fake_index)

        chunks = ["hello world", "second chunk"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        VectorService.upsert_vectors(embeddings, chunks, "doc.pdf", page_numbers=[2, 5])

        vecs = captured["vectors"]
        assert vecs[0]["metadata"]["page_no"] == 2
        assert vecs[1]["metadata"]["page_no"] == 5
        assert (
            vecs[0]["metadata"]["content_hash"]
            == hashlib.sha256(chunks[0].encode()).hexdigest()
        )
        assert (
            vecs[1]["metadata"]["content_hash"]
            == hashlib.sha256(chunks[1].encode()).hexdigest()
        )
        assert vecs[0]["metadata"]["content"] == chunks[0]
        assert vecs[0]["metadata"]["chunk_index"] == 0

    def test_upsert_without_page_numbers_still_hashes(self, monkeypatch):
        """Do test upsert without page numbers still hashes."""
        import hashlib

        import config

        captured = {}

        def fake_upsert(self, vectors):
            """Do fake upsert."""
            captured["vectors"] = vectors
            return {}

        fake_index = type("Idx", (), {"upsert": fake_upsert})()
        monkeypatch.setattr(config, "_pinecone_index", fake_index)

        chunks = ["hello"]
        VectorService.upsert_vectors([[0.1]], chunks, "doc.pdf")

        assert (
            captured["vectors"][0]["metadata"]["content_hash"]
            == hashlib.sha256(b"hello").hexdigest()
        )
        assert captured["vectors"][0]["metadata"]["page_no"] is None

    def test_matches_to_sources_preserves_page_no_and_hash(self):
        """Do test matches to sources preserves page no and hash."""
        from services.vector_service import matches_to_sources

        matches = [
            {
                "score": 0.9,
                "metadata": {
                    "content": "some content",
                    "pdf_name": "doc.pdf",
                    "chunk_index": 3,
                    "page_no": 7,
                    "content_hash": "abc123",
                },
            }
        ]
        sources = matches_to_sources(matches, "doc.pdf")
        assert sources[0]["page_no"] == 7
        assert sources[0]["content_hash"] == "abc123"
        assert sources[0]["chunk_index"] == 3

    def test_shape_sources_keeps_metadata_through_dedupe(self, fake_index):
        # Highest scoring duplicate should keep its page_no/hash
        """Do test shape sources keeps metadata through dedupe."""
        matches = [
            {
                "id": "v-low",
                "score": 0.4,
                "metadata": {
                    "content": "same text",
                    "pdf_name": "doc.pdf",
                    "chunk_index": 0,
                    "page_no": 1,
                    "content_hash": "hash1",
                },
            },
            {
                "id": "v-high",
                "score": 0.9,
                "metadata": {
                    "content": "same text",
                    "pdf_name": "doc.pdf",
                    "chunk_index": 1,
                    "page_no": 5,
                    "content_hash": "hash2",
                },
            },
        ]
        sources, _ = TestRetrievalShaping().run_shaping(fake_index, matches)
        assert len(sources) == 1
        assert sources[0]["page_no"] == 5
        assert sources[0]["content_hash"] == "hash2"
