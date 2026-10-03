"""Unit tests for shared route helpers.

Traversal guards run against a real LocalStorage in a temp dir. Flask-only
helpers use a bare app context. Run with plain ``pytest``.
"""

import pytest

from errors import AppError, Conflict
from routes.common import (
    check_filename,
    deletion_blocked,
    file_url,
    is_safe_filename,
    raise_refusal,
    raise_scope_refusal,
    scrub_api_key_from_text,
    stream_response,
)
from services.streaming import AnswerEvent, Refusal
from storage import LocalStorage

pytestmark = pytest.mark.unit


@pytest.fixture()
def storage(tmp_path):
    return LocalStorage(tmp_path / "storage")


class TestCheckFilename:
    def test_plain_name_passes(self, storage):
        check_filename(storage, "abc123.pdf")

    @pytest.mark.parametrize(
        "name", ["../secret.pdf", "..\\secret.pdf", "/etc/passwd", "\\abs.pdf"]
    )
    def test_traversal_rejected(self, storage, name):
        with pytest.raises(AppError) as exc_info:
            check_filename(storage, name)
        assert exc_info.value.category == "invalid_filename"

    def test_plain_dict_storage_fallback(self):
        check_filename(object(), "doc.pdf")
        with pytest.raises(AppError):
            check_filename(object(), "../doc.pdf")

    def test_broken_path_probe_falls_back(self):
        class WeirdStorage:
            def _path(self, name):
                raise RuntimeError("weird")

        check_filename(WeirdStorage(), "doc.pdf")
        with pytest.raises(AppError):
            check_filename(WeirdStorage(), "../doc.pdf")


class TestIsSafeFilename:
    def test_safe(self, storage):
        assert is_safe_filename(storage, "doc.pdf") is True

    def test_unsafe(self, storage):
        assert is_safe_filename(storage, "../doc.pdf") is False

    def test_broken_probe_treats_dots_as_unsafe(self):
        class WeirdStorage:
            def _path(self, name):
                raise RuntimeError("weird")

        assert is_safe_filename(WeirdStorage(), "doc.pdf") is True
        assert is_safe_filename(WeirdStorage(), "../doc.pdf") is False


class TestRaiseRefusal:
    def test_non_refusal_returned(self):
        assert raise_refusal({"ok": True}) == {"ok": True}

    def test_refusal_raised_as_error(self):
        with pytest.raises(AppError) as exc_info:
            raise_refusal(Refusal(400, "invalid_query", "bad query"))
        assert exc_info.value.category == "invalid_query"
        assert exc_info.value.status == 400

    def test_scope_refusal_carries_extras(self):
        with pytest.raises(AppError) as exc_info:
            raise_scope_refusal(
                {
                    "status": 409,
                    "category": "index_stale",
                    "error": "stale",
                    "label": "A",
                }
            )
        assert exc_info.value.status == 409
        assert exc_info.value.details["label"] == "A"

    def test_scope_refusal_defaults(self):
        with pytest.raises(AppError) as exc_info:
            raise_scope_refusal(None)
        assert exc_info.value.status == 400


class TestDeletionBlocked:
    def test_mid_deletion(self):
        error = deletion_blocked({"deletion_state": "deleting"})
        assert isinstance(error, Conflict)
        assert error.category == "document_deleting"

    def test_failed_deletion(self):
        error = deletion_blocked(
            {"deletion_state": "delete_failed", "deletion_error": "x"}
        )
        assert error.category == "document_delete_failed"
        assert error.details["deletion_error"] == "x"


class TestScrubApiKey:
    def test_full_key_and_fragments_removed(self):
        key = "sk-abcdefghijklmnop"
        scrubbed = scrub_api_key_from_text(f"call with {key} inside", key)
        assert key not in scrubbed
        assert "••••" in scrubbed

    def test_url_encoded_form_removed(self):
        import urllib.parse

        key = "sk-abc def/ghi+jkl"
        encoded = urllib.parse.quote(key, safe="")
        scrubbed = scrub_api_key_from_text(f"token {encoded} here", key)
        assert encoded not in scrubbed

    def test_empty_inputs(self):
        assert scrub_api_key_from_text(None, "key") is None
        assert scrub_api_key_from_text("text", None) == "text"
        assert scrub_api_key_from_text("", "key") == ""


class TestFlaskHelpers:
    def test_file_url_needs_request_context(self, storage):
        from flask import Flask

        app = Flask(__name__)
        with app.test_request_context("/", base_url="http://localhost:3000"):
            assert file_url(storage, "a.pdf") == "http://localhost:3000/storage/a.pdf"

    def test_stream_response_framing(self):
        from flask import Flask

        app = Flask(__name__)
        events = [
            AnswerEvent("token", {"text": "hi"}),
            AnswerEvent("done", {"done": True}),
        ]
        with app.test_request_context("/"):
            response = stream_response(iter(events))
            assert response.mimetype == "text/event-stream"
            body = "".join(response.response)
        assert "event: token" in body
        assert '"text": "hi"' in body
        assert "event: done" in body
