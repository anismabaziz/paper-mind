"""
Tests for the document parser and retrieval shaping.

The parser owns extraction and chunking; the PDF implementation is one
parser among future ones. Retrieval shaping turns raw vector matches into a
deduped, score-ordered, bounded source list. Both run against fakes or
in-memory data only.
"""

import pytest

from services.parsing.document_parser import (
    Chunk,
    DocumentIngestor,
    TokenChunker,
    UnknownDocumentFormat,
    resolve_parser,
)
from services.retrieval.hybrid import (
    SPARSE_METHOD,
    TOKENIZER_VERSION,
    build_sparse_vector,
)
from services.retrieval.vector_service import (
    MAX_RETRIEVED_SOURCES,
    VectorService,
    build_vectors_from_chunks,
)


class TestDocumentParser:
    """TestDocumentParser."""

    def test_pdf_filename_resolves_the_pdf_parser(self):
        """Do test pdf filename resolves the pdf parser."""
        from services.parsing.pdf_service import PDFParser

        assert isinstance(resolve_parser("paper.pdf"), PDFParser)

    def test_unknown_format_is_rejected_with_a_readable_error(self):
        """Do test unknown format is rejected with a readable error."""
        with pytest.raises(UnknownDocumentFormat) as exc:
            resolve_parser("scan.docx")
        assert ".docx" in str(exc.value)

    def test_pdf_extraction_normalizes_whitespace_per_page(self):
        """Do test pdf extraction normalizes whitespace per page."""
        parser = resolve_parser("paper.pdf")
        text = parser.extract_text(self._two_page_pdf())

        assert "first page words" in text
        assert "second page words" in text
        assert "\n" not in text, "page text must be normalized before joining"
        assert "  " not in text

    def test_split_text_bounds_chunks_by_token_count(self):
        """Do test split text bounds chunks by token count."""
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        text = " ".join(f"word{i}" for i in range(3000))

        chunks = TokenChunker(512, 50).split_text(text)

        assert len(chunks) > 1
        # Token-bounded: each chunk is ~512 tokens; last may be smaller.
        for chunk in chunks[:-1]:
            n = len(enc.encode(chunk))
            assert 487 <= n <= 537, f"chunk {n} tokens outside 512±25"
        assert len(enc.encode(chunks[-1])) <= 537
        # Overlap preservation: ~50 tokens across boundary (token-level check).
        t0 = enc.encode(chunks[0])
        t1 = enc.encode(chunks[1])
        tail_tokens = set(t0[-50:])
        head_tokens = set(t1[:60])
        assert len(tail_tokens & head_tokens) >= 15, (
            "overlap should preserve ~50 tokens across boundary"
        )

    def test_ingestor_chunks_with_the_settings_values_it_was_built_with(self):
        """The ingestor's chunking matches a chunker built with the same values."""
        text = " ".join(f"word{i}" for i in range(3000))

        ingestor = DocumentIngestor(
            use_docling="auto", chunk_size=512, chunk_overlap=50
        )

        assert ingestor.split_text(text) == TokenChunker(512, 50).split_text(text)

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

        parser = resolve_parser("paper.pdf")
        pages = parser.extract_pages(pdf_bytes)
        assert len(pages) == 2
        chunks_with_page = TokenChunker(512, 50).split_pages(pages)
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

        parser = resolve_parser("paper.pdf")
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


def test_sparse_vector_aggregates_hash_collisions():
    """Colliding token hashes produce one weighted sparse index."""
    vector = build_sparse_vector("acc acr")

    assert vector == {"indices": [1750], "values": [2.0]}


def test_query_without_sparse_terms_selects_dense_explicitly():
    """A stopword-only query records the dense method it selected."""
    index = FakeVectorIndex([])

    result = VectorService(index).query_vectors([0.1], "doc.pdf", query_text="the and")

    assert result.method == "dense"
    assert result.outcome == "empty"
    assert index.queries[0]["kwargs"]["method"] == "dense"
    assert "sparse_vector" not in index.queries[0]["kwargs"]


def test_service_records_the_selected_retrieval_method():
    """A shaped result names the retrieval method that ran."""
    index = FakeVectorIndex(
        [
            {
                "id": "point-1",
                "score": 0.5,
                "metadata": {
                    "content": "keyword evidence",
                    "pdf_name": "doc.pdf",
                    "chunk_index": 0,
                },
            }
        ]
    )

    result = VectorService(index).query_vectors(
        [0.1], "doc.pdf", query_text="keyword", method="sparse"
    )

    assert result.method == "sparse"
    assert result.outcome == "success"
    assert result.sources[0]["content"] == "keyword evidence"
    assert index.queries[0]["kwargs"]["method"] == "sparse"


class FakeVectorIndex:
    """FakeVectorIndex."""

    def __init__(self, matches):
        """Initialize."""
        self._matches = matches
        self.queries = []

    def query(self, vector, top_k, include_metadata, filter, **kwargs):
        """Do query."""
        self.queries.append({"top_k": top_k, "filter": filter, "kwargs": kwargs})
        method = kwargs.get("method", "dense")
        return {
            "matches": self._matches,
            "method": method,
            "outcome": "success" if self._matches else "empty",
        }


@pytest.fixture
def service_factory():
    """Build a VectorService over a fake index; both are returned."""

    def install(matches):
        """Do install."""
        index = FakeVectorIndex(matches)
        return VectorService(index), index

    return install


class TestRetrievalShaping:
    """TestRetrievalShaping."""

    def run_shaping(self, service_factory, matches):
        """Do run shaping."""
        service, index = service_factory(matches)
        return service.query_vectors([0.1], "doc.pdf").sources, index

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

    def test_results_are_ordered_by_score_descending(self, service_factory):
        """Do test results are ordered by score descending."""
        matches = [
            self.match("low", 0.10, 0),
            self.match("high", 0.90, 1),
            self.match("mid", 0.50, 2),
        ]

        sources, _ = self.run_shaping(service_factory, matches)

        assert [s["content"] for s in sources] == ["high", "mid", "low"]

    def test_duplicate_content_is_deduped_keeping_the_best_score(self, service_factory):
        """Do test duplicate content is deduped keeping the best score."""
        matches = [
            self.match("same text", 0.40, 0),
            self.match("same text", 0.80, 1),
            self.match("unique", 0.60, 2),
        ]

        sources, _ = self.run_shaping(service_factory, matches)

        assert [s["content"] for s in sources] == ["same text", "unique"]
        assert sources[0]["score"] == 0.80

    def test_results_are_bounded(self, service_factory):
        """Do test results are bounded."""
        matches = [self.match(f"chunk {i}", 1.0 - i / 10, i) for i in range(10)]

        sources, _ = self.run_shaping(service_factory, matches)

        assert len(sources) == MAX_RETRIEVED_SOURCES
        assert len(sources) < len(matches)

    def test_matches_without_content_are_dropped(self, service_factory):
        """Do test matches without content are dropped."""
        matches = [
            {"id": "v-empty", "score": 0.9, "metadata": {"pdf_name": "doc.pdf"}},
            self.match("real", 0.5, 0),
        ]

        sources, _ = self.run_shaping(service_factory, matches)

        assert [s["content"] for s in sources] == ["real"]

    def test_query_is_filtered_to_the_document(self, service_factory):
        """Do test query is filtered to the document."""
        _, index = service_factory([])

        VectorService(index).query_vectors([0.1], "doc.pdf")

        assert index.queries[0]["filter"] == {"pdf_name": "doc.pdf"}


class TestChunkMetadata:
    """TestChunkMetadata."""

    def test_shared_builder_preserves_sparse_and_chunk_metadata(self):
        """A batch offset does not change sparse text or source metadata."""
        chunks = [
            Chunk("hello", page_no=2, chunk_index=0, content_hash="hello-hash"),
            Chunk("world", page_no=7, chunk_index=1, content_hash="world-hash"),
        ]

        vectors = build_vectors_from_chunks([[0.3, 0.4]], chunks, "paper.pdf", offset=1)

        assert len(vectors) == 1
        vector = vectors[0]
        assert vector["id"]
        assert vector["values"] == [0.3, 0.4]
        assert vector["sparse_vector"] == {"indices": [23351], "values": [1.0]}
        assert vector["metadata"] == {
            "content": "world",
            "pdf_name": "paper.pdf",
            "chunk_index": 1,
            "page_no": 7,
            "content_hash": "world-hash",
            "sparse_method": SPARSE_METHOD,
            "sparse_tokenizer_version": TOKENIZER_VERSION,
        }

    def test_generation_point_ids_are_deterministic_and_isolated(self):
        """The same generation reuses point IDs while a new generation does not."""
        chunks = [Chunk("hello", page_no=1, chunk_index=0, content_hash="hash")]

        first = build_vectors_from_chunks([[0.3]], chunks, "paper.pdf", generation=1)
        repeated = build_vectors_from_chunks([[0.3]], chunks, "paper.pdf", generation=1)
        replacement = build_vectors_from_chunks(
            [[0.3]], chunks, "paper.pdf", generation=2
        )

        assert first[0]["id"] == repeated[0]["id"]
        assert first[0]["id"] != replacement[0]["id"]
        assert first[0]["metadata"]["index_generation"] == 1

    def test_generation_validation_rejects_an_incomplete_index(self):
        """A generation with the wrong point count cannot become ready."""
        from services.retrieval.base import VectorStoreConfigurationError

        class CountingStore:
            """Store double that reports an incomplete generation."""

            def count(self, filter=None):
                return 1

        with pytest.raises(VectorStoreConfigurationError):
            VectorService(CountingStore()).validate_generation("doc.pdf", 2, 2)

    def test_upsert_includes_page_no_and_content_hash(self):
        """Do test upsert includes page no and content hash."""
        import hashlib

        captured = {}

        def fake_upsert(self, vectors):
            """Do fake upsert."""
            captured["vectors"] = vectors
            return {"upserted": len(vectors)}

        index = type("Idx", (), {"upsert": fake_upsert})()

        chunks = ["hello world", "second chunk"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        VectorService(index).upsert_vectors(
            embeddings, chunks, "doc.pdf", page_numbers=[2, 5]
        )

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

    def test_upsert_without_page_numbers_still_hashes(self):
        """Do test upsert without page numbers still hashes."""
        import hashlib

        captured = {}

        def fake_upsert(self, vectors):
            """Do fake upsert."""
            captured["vectors"] = vectors
            return {}

        index = type("Idx", (), {"upsert": fake_upsert})()

        chunks = ["hello"]
        VectorService(index).upsert_vectors([[0.1]], chunks, "doc.pdf")

        assert (
            captured["vectors"][0]["metadata"]["content_hash"]
            == hashlib.sha256(b"hello").hexdigest()
        )
        assert captured["vectors"][0]["metadata"]["page_no"] is None

    def test_matches_to_sources_preserves_page_no_and_hash(self):
        """Do test matches to sources preserves page no and hash."""
        from services.retrieval.vector_service import matches_to_sources

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

    def test_shape_sources_keeps_metadata_through_dedupe(self, service_factory):
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
        sources, _ = TestRetrievalShaping().run_shaping(service_factory, matches)
        assert len(sources) == 1
        assert sources[0]["page_no"] == 5
        assert sources[0]["content_hash"] == "hash2"
