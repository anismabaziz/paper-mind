"""Integration fixtures: Flask app over fakes, no network or DB.

Run with ``pytest tests/integration``. Never collected by plain ``pytest``.
"""

import copy
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import settings as settings_module
from repositories.ingestion_jobs import JobConflictError
from services.accounts.chat_settings_service import model_for
from services.accounts.secrets_service import encrypt_api_key
from services.indexing.manifest import runtime_manifest
from services.retrieval.base import RetrievalResult
from services.telemetry.spans import Tracer


@pytest.fixture()
def isettings(tmp_path):
    """Test settings isolated per test and installed globally.

    Installed globally because key encryption reads the process settings.
    Restored after the test so unit runs cannot leak.
    """
    try:
        previous = settings_module.get_settings()
    except Exception:
        previous = None
    candidate = settings_module.Settings()
    candidate.storage.storage_dir = tmp_path / "storage"
    candidate.database.database_url = "postgresql://test:test@localhost:5432/test"
    candidate.auth.app_secret = "integration-test-secret"
    settings_module.set_settings(candidate)
    yield candidate
    settings_module.set_settings(previous)


def make_file_record(**overrides):
    record = {
        "id": "file-1",
        "filename": "doc1.pdf",
        "title": "Test Paper",
        "original_filename": "test-paper.pdf",
        "is_processed": False,
        "index_generation": 0,
        "index_manifest": None,
        "index_stale_reason": None,
        "deletion_state": "active",
        "deletion_error": None,
        "deletion_attempts": 0,
        "last_opened_at": None,
    }
    record.update(overrides)
    return copy.deepcopy(record)


def make_ready_record(settings, **overrides):
    manifest = runtime_manifest(
        settings, content_hash_value="hash", index_generation=1
    ).to_json()
    record = make_file_record(
        is_processed=True, index_generation=1, index_manifest=manifest
    )
    record.update(overrides)
    return copy.deepcopy(record)


def make_pdf_bytes(title="Test Paper", text="Hello world content"):
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.set_metadata({"title": title})
    buffer = io.BytesIO()
    doc.save(buffer)
    doc.close()
    return buffer.getvalue()


class FakeFiles:
    def __init__(self, records=()):
        self._records = {r["filename"]: copy.deepcopy(r) for r in records}
        self.titles = {}
        self.stale = []
        self.deleted_ids = []

    def get_file(self, filename):
        record = self._records.get(filename)
        return copy.deepcopy(record) if record else None

    def list_files(self):
        return [copy.deepcopy(r) for r in self._records.values()]

    def create_file_with_job(self, filename, title=None, original_filename=None):
        record = make_file_record(
            id=f"file-{filename}",
            filename=filename,
            title=title,
            original_filename=original_filename,
        )
        self._records[filename] = copy.deepcopy(record)
        return copy.deepcopy(record), {"id": "job-1", "state": "queued"}

    def touch_opened(self, filename):
        if filename not in self._records:
            return None
        self._records[filename]["last_opened_at"] = "2026-01-01T00:00:00"
        return {"last_opened_at": "2026-01-01T00:00:00"}

    def mark_deleting(self, filename):
        if filename not in self._records:
            return None
        self._records[filename]["deletion_state"] = "deleting"
        return copy.deepcopy(self._records[filename])

    def mark_delete_failed(self, filename, reason):
        if filename in self._records:
            self._records[filename]["deletion_state"] = "delete_failed"
            self._records[filename]["deletion_error"] = reason

    def delete_file(self, file_id):
        self.deleted_ids.append(file_id)
        self._records = {
            name: r for name, r in self._records.items() if r["id"] != file_id
        }

    def set_index_stale(self, filename, reason):
        self.stale.append((filename, reason))

    def set_file_title(self, filename, title):
        self.titles[filename] = title
        if filename in self._records:
            self._records[filename]["title"] = title


class FakeConversations:
    def __init__(self, conversation_id="conv-1", messages=()):
        self._conversation_id = conversation_id
        self._messages = list(messages)
        self.calls = []

    def get_conversation_id(self, file_id):
        return self._conversation_id

    def get_messages(self, conversation_id):
        return list(self._messages)

    def get_recent_turns(self, conversation_id, limit):
        return []

    def count_answered_turns(self, conversation_id):
        return 0

    def start_turn(self, conversation_id, query):
        self.calls.append(("start", query))
        return "turn-1"

    def complete_turn(self, turn_id, answer, sources, claims=()):
        self.calls.append(("complete", turn_id, answer))
        return True

    def fail_turn(self, turn_id, message, reason):
        self.calls.append(("fail", turn_id, reason))
        return True

    def cancel_turn(self, turn_id, reason):
        self.calls.append(("cancel", turn_id, reason))
        return True

    def abstain_turn(self, turn_id, message, reason):
        self.calls.append(("abstain", turn_id, reason))
        return True

    def get_conversation_id_for_delete(self, file_id):
        return self._conversation_id

    def delete_conversation_tree(self, conversation_id):
        self.calls.append(("delete_tree", conversation_id))


class FakeIngestionJobs:
    def __init__(self, latest=None, active=None, conflict=False):
        self._latest = latest
        self._active = active
        self._conflict = conflict
        self.cancelled = []
        self.deleted = []

    def get_latest(self, filename):
        return copy.deepcopy(self._latest) if self._latest else None

    def get_active(self, file_id):
        return copy.deepcopy(self._active) if self._active else None

    def enqueue(self, file_id, filename):
        if self._conflict:
            raise JobConflictError("already active")
        job = {"id": "job-9", "state": "queued"}
        self._active = copy.deepcopy(job)
        return job

    def request_cancel(self, file_id):
        return copy.deepcopy(self._latest) if self._latest else None

    def cancel_active(self, file_id):
        self.cancelled.append(file_id)
        return 1

    def delete_for_file(self, file_id):
        self.deleted.append(file_id)


class FakeCleanups:
    def __init__(self):
        self.deleted = []

    def delete_for_file(self, file_id):
        self.deleted.append(file_id)

    def schedule(self, *args):
        pass


class FakeAppSettings:
    def __init__(self, stored=None):
        self._stored = stored
        self.upserts = []

    def get_app_settings(self):
        return copy.deepcopy(self._stored) if self._stored else None

    def upsert_app_settings(self, provider, model, ciphertext):
        self.upserts.append((provider, model))
        self._stored = {
            "provider": provider,
            "model": model,
            "encrypted_api_key": ciphertext,
        }
        return copy.deepcopy(self._stored)


class FakeVectorService:
    def __init__(self, result=None):
        self._result = result or RetrievalResult(
            sources=[], method="dense", outcome="empty"
        )
        self.deleted_names = []
        self.deleted_all = 0

    def retrieve(self, query, filename, generation=None, include_legacy=False):
        return self._result

    def delete_by_filename(self, filename):
        self.deleted_names.append(filename)

    def delete_all(self):
        self.deleted_all += 1


class FakeProvider:
    def __init__(self, tokens=(), repair="fixed", last_finish_reason="stop"):
        self._tokens = list(tokens)
        self._repair = repair
        self.last_finish_reason = last_finish_reason
        self.last_attempts = 1

    def stream_response(self, query, context, prior_turns=""):
        yield from self._tokens

    def generate_response(self, query, context, prior_turns=""):
        return self._repair


def make_services(
    settings,
    files=None,
    conversations=None,
    jobs=None,
    app_settings=None,
    vectors=None,
    provider=None,
    verifier=None,
):
    """Build a Services graph where every external call hits a fake."""
    from composition import Services
    from storage import LocalStorage

    provider = provider or FakeProvider()
    return Services(
        settings=settings,
        repositories=SimpleNamespace(
            files=files or FakeFiles(),
            app_settings=app_settings or FakeAppSettings(),
            conversations=conversations or FakeConversations(),
            ingestion_jobs=jobs or FakeIngestionJobs(),
            index_cleanups=FakeCleanups(),
        ),
        storage=LocalStorage(settings.storage.storage_dir),
        parser=object(),
        embedding_service=object(),
        vector_service=vectors or FakeVectorService(),
        chat_provider_factory=lambda credentials: provider,
        api_key_verifier=verifier or (lambda credentials: (True, None)),
        tracer=Tracer(),
    )


def make_client(services, settings):
    from app import create_app

    app = create_app(app_settings=settings, services=services)
    return app.test_client()


@pytest.fixture()
def provider_model():
    return model_for("google", "gemini-2.5-flash")


@pytest.fixture()
def stored_settings(isettings):
    """An app-settings row with a really encrypted key for the test secret."""
    return {
        "provider": "google",
        "model": "gemini-2.5-flash",
        "encrypted_api_key": encrypt_api_key("test-api-key"),
    }


__all__ = ["JobConflictError"]
