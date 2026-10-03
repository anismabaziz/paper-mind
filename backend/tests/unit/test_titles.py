"""Unit tests for document title derivation and backfill.

Title parsing uses the real local PDF reader on tiny generated PDFs.
Run with plain ``pytest``.
"""

import io

import pytest

from services.titles import backfill_title, derive_title, is_hex_like_title

pytestmark = pytest.mark.unit


def _pdf_bytes(metadata_title=None, page_text="First Heading Line"):
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), page_text)
    if metadata_title is not None:
        doc.set_metadata({"title": metadata_title})
    buffer = io.BytesIO()
    doc.save(buffer)
    doc.close()
    return buffer.getvalue()


class TestIsHexLikeTitle:
    def test_hex_name(self):
        assert is_hex_like_title("a" * 32) is True

    def test_hex_with_extension(self):
        assert is_hex_like_title(f"{'b' * 32}.pdf") is True

    def test_human_title(self):
        assert is_hex_like_title("Attention Is All You Need") is False

    def test_none_and_empty(self):
        assert is_hex_like_title(None) is False
        assert is_hex_like_title("") is False
        assert is_hex_like_title("   ") is False

    def test_near_miss_lengths(self):
        assert is_hex_like_title("a" * 31) is False
        assert is_hex_like_title("a" * 33) is False


class TestDeriveTitle:
    def test_metadata_title_wins(self):
        pdf = _pdf_bytes(metadata_title="  Real Paper Title  ")
        assert derive_title(pdf, "upload.pdf") == "Real Paper Title"

    def test_blank_metadata_falls_to_filename(self):
        pdf = _pdf_bytes(metadata_title="   ")
        assert derive_title(pdf, "my-paper.pdf") == "my-paper"

    def test_no_bytes_uses_filename(self):
        assert derive_title(None, "my-paper.pdf") == "my-paper"

    def test_filename_stripped_of_dirs_and_extension(self):
        assert derive_title(None, "uploads/My Paper v2.pdf") == "My Paper v2"

    def test_no_filename_uses_heading(self):
        pdf = _pdf_bytes()
        assert derive_title(pdf, None) == "First Heading Line"

    def test_nothing_gives_untitled(self):
        assert derive_title(None, None) == "Untitled"
        assert derive_title(b"", "") == "Untitled"

    def test_garbage_bytes_give_untitled(self):
        assert derive_title(b"not a pdf at all", None) == "Untitled"

    def test_bytearray_accepted(self):
        pdf = bytearray(_pdf_bytes(metadata_title="Byte Title"))
        assert derive_title(pdf, None) == "Byte Title"


class FakeStorage:
    def __init__(self, payload=None, error=None):
        self._payload = payload
        self._error = error

    def open(self, filename):
        if self._error is not None:
            raise self._error
        return self._payload


class FakeRepository:
    def __init__(self):
        self.saved = []

    def set_file_title(self, filename, title):
        self.saved.append((filename, title))


class TestBackfillTitle:
    def test_human_title_untouched(self):
        repo = FakeRepository()
        out = backfill_title(FakeStorage(), repo, "f.pdf", "Real Title", "real.pdf")
        assert out == "Real Title"
        assert repo.saved == []

    def test_hex_title_backfilled_and_persisted(self):
        repo = FakeRepository()
        pdf = _pdf_bytes(metadata_title="Derived Title")
        out = backfill_title(FakeStorage(pdf), repo, "f.pdf", "c" * 32, None)
        assert out == "Derived Title"
        assert repo.saved == [("f.pdf", "Derived Title")]

    def test_missing_title_backfilled(self):
        repo = FakeRepository()
        out = backfill_title(FakeStorage(None), repo, "f.pdf", None, "named.pdf")
        assert out == "named"
        assert repo.saved == [("f.pdf", "named")]

    def test_unreadable_storage_keeps_title(self):
        repo = FakeRepository()
        out = backfill_title(
            FakeStorage(error=OSError("gone")), repo, "f.pdf", None, "named.pdf"
        )
        assert out is None
        assert repo.saved == []

    def test_persist_failure_still_returns_derived(self):
        class BadRepo:
            def set_file_title(self, filename, title):
                raise OSError("db down")

        out = backfill_title(FakeStorage(None), BadRepo(), "f.pdf", None, "named.pdf")
        assert out == "named"
