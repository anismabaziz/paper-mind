"""Slow tests for repositories against real Postgres.

Needs the ``papermind_test`` database (``pytest tests/slow`` skips without
it). Tables are emptied after each test. Covers the queries the routes and
the worker actually run: files, conversations with turns, ingestion jobs,
app settings, and generation cleanups.
"""

import pytest

pytestmark = pytest.mark.slow


class TestFiles:
    def test_create_and_read(self, repositories):
        record, job = repositories.files.create_file_with_job(
            "a.pdf", title="Paper", original_filename="paper.pdf"
        )
        assert record["filename"] == "a.pdf"
        assert job["state"] == "queued"
        fetched = repositories.files.get_file("a.pdf")
        assert fetched["id"] == record["id"]
        assert fetched["title"] == "Paper"

    def test_list_and_titles(self, repositories):
        repositories.files.create_file("a.pdf", title="A")
        repositories.files.create_file("b.pdf", title="B")
        assert len(repositories.files.list_files()) == 2
        repositories.files.set_file_title("a.pdf", "Alpha")
        assert repositories.files.get_file("a.pdf")["title"] == "Alpha"

    def test_delete_lifecycle(self, repositories):
        record, _ = repositories.files.create_file_with_job("a.pdf")
        assert repositories.files.mark_deleting("a.pdf")["deletion_state"] == "deleting"
        repositories.files.mark_delete_failed("a.pdf", "vectors stuck")
        assert repositories.files.get_file("a.pdf")["deletion_state"] == "delete_failed"
        repositories.files.delete_file(record["id"])
        assert repositories.files.get_file("a.pdf") is None

    def test_processed_and_stale_flags(self, repositories):
        repositories.files.create_file("a.pdf")
        repositories.files.set_processed("a.pdf", True)
        repositories.files.set_index_stale("a.pdf", "chunk size")
        fetched = repositories.files.get_file("a.pdf")
        assert fetched["is_processed"] is True
        assert fetched["index_stale_reason"] == "chunk size"
        assert repositories.files.touch_opened("a.pdf")["last_opened_at"] is not None


class TestConversations:
    def _conversation(self, repositories):
        record, _ = repositories.files.create_file_with_job("a.pdf")
        return repositories.conversations.ensure_conversation(record["id"])

    def test_turn_lifecycle(self, repositories):
        conversation_id = self._conversation(repositories)
        assert (
            repositories.conversations.get_conversation_id(
                repositories.files.get_file("a.pdf")["id"]
            )
            == conversation_id
        )
        turn_id = repositories.conversations.start_turn(conversation_id, "What is X?")
        assert (
            repositories.conversations.complete_turn(
                turn_id,
                "X is Y.",
                [
                    {
                        "content": "X is Y.",
                        "document": "a.pdf",
                        "chunk_index": 0,
                        "score": 1.0,
                        "page": 1,
                    }
                ],
                claims=[{"claim": "X is Y", "sources": ["S1"]}],
            )
            is True
        )
        messages = repositories.conversations.get_messages(conversation_id)
        assert [m["sender"] for m in messages] == ["user", "bot"]
        assert messages[1]["turn_status"] == "answered"
        assert messages[1]["claims"] == [{"claim": "X is Y", "sources": ["S1"]}]

    def test_fail_and_cancel(self, repositories):
        conversation_id = self._conversation(repositories)
        failed = repositories.conversations.start_turn(conversation_id, "q1")
        assert repositories.conversations.fail_turn(failed, "down", "provider") is True
        cancelled = repositories.conversations.start_turn(conversation_id, "q2")
        assert repositories.conversations.cancel_turn(cancelled, "gone") is True
        assert repositories.conversations.count_answered_turns(conversation_id) == 0

    def test_abstain_and_recent(self, repositories):
        conversation_id = self._conversation(repositories)
        answered = repositories.conversations.start_turn(conversation_id, "q1")
        repositories.conversations.complete_turn(answered, "a1", [])
        assert len(repositories.conversations.get_recent_turns(conversation_id, 5)) == 1
        abstained = repositories.conversations.start_turn(conversation_id, "q2")
        assert (
            repositories.conversations.abstain_turn(
                abstained, "no evidence", "no_evidence"
            )
            is True
        )
        recent = repositories.conversations.get_recent_turns(conversation_id, 5)
        assert len(recent) == 1
        assert recent[0]["question"] == "q1"

    def test_delete_tree(self, repositories):
        conversation_id = self._conversation(repositories)
        turn_id = repositories.conversations.start_turn(conversation_id, "q?")
        repositories.conversations.complete_turn(turn_id, "a", [])
        repositories.conversations.delete_conversation_tree(conversation_id)
        assert repositories.conversations.get_messages(conversation_id) == []


class TestIngestionJobs:
    def _file(self, repositories, name="a.pdf"):
        record, _ = repositories.files.create_file_with_job(name)
        return record

    def test_enqueue_conflicts_with_running(self, repositories):
        from repositories.ingestion_jobs import JobConflictError

        record = self._file(repositories)
        repositories.ingestion_jobs.claim_next("worker-1", 120)
        with pytest.raises(JobConflictError):
            repositories.ingestion_jobs.enqueue(record["id"], record["filename"])

    def test_enqueue_supersedes_queued(self, repositories):
        record = self._file(repositories)
        second = repositories.ingestion_jobs.enqueue(record["id"], record["filename"])
        assert second["generation"] == 2
        assert (
            repositories.ingestion_jobs.get_active(record["id"])["id"] == second["id"]
        )

    def test_claim_and_ready(self, repositories):
        record = self._file(repositories)
        claimed = repositories.ingestion_jobs.claim_next("worker-1", 120)
        assert claimed is not None
        assert claimed["state"] == "running"
        ready = repositories.ingestion_jobs.mark_ready(
            claimed["id"], "worker-1", index_generation=1, index_manifest="{}"
        )
        assert ready["state"] == "ready"
        assert repositories.files.get_file("a.pdf")["is_processed"] is True

    def test_cancel_queued_finishes_immediately(self, repositories):
        record = self._file(repositories)
        cancelled = repositories.ingestion_jobs.request_cancel(record["id"])
        assert cancelled is not None
        assert cancelled["state"] == "cancelled"
        assert repositories.ingestion_jobs.get_active(record["id"]) is None

    def test_cancel_running_moves_to_cancelling(self, repositories):
        record = self._file(repositories)
        repositories.ingestion_jobs.claim_next("worker-1", 120)
        requested = repositories.ingestion_jobs.request_cancel(record["id"])
        assert requested is not None
        assert requested["state"] == "cancelling"
        assert requested["cancel_requested_at"] is not None
        assert repositories.ingestion_jobs.get_active(record["id"]) is not None
        repositories.ingestion_jobs.cancel_active(record["id"])
        assert (
            repositories.ingestion_jobs.get_active(record["id"])["state"]
            == "cancelling"
        )

    def test_fail_and_delete(self, repositories):
        record = self._file(repositories)
        claimed = repositories.ingestion_jobs.claim_next("worker-1", 120)
        failed = repositories.ingestion_jobs.mark_failed(
            claimed["id"], "worker-1", "page_limit_exceeded", "too big"
        )
        assert failed is not None
        assert failed["state"] == "failed"
        assert failed["error_category"] == "page_limit_exceeded"
        latest = repositories.ingestion_jobs.get_latest("a.pdf")
        assert latest["state"] == "failed"
        repositories.ingestion_jobs.delete_for_file(record["id"])
        assert repositories.ingestion_jobs.get_latest("a.pdf") is None


class TestAppSettingsAndCleanups:
    def test_settings_upsert(self, repositories):
        assert repositories.app_settings.get_app_settings() is None
        repositories.app_settings.upsert_app_settings(
            "google", "gemini-2.5-flash", "ct"
        )
        stored = repositories.app_settings.get_app_settings()
        assert stored["provider"] == "google"
        repositories.app_settings.upsert_app_settings(
            "groq", "openai/gpt-oss-20b", "ct2"
        )
        assert repositories.app_settings.get_app_settings()["provider"] == "groq"

    def test_cleanup_lifecycle(self, repositories):
        record, _ = repositories.files.create_file_with_job("a.pdf")
        repositories.index_cleanups.schedule(record["id"], "a.pdf", 1)
        (pending,) = repositories.index_cleanups.list_pending()
        assert pending["generation"] == 1
        repositories.index_cleanups.mark_done(pending["id"])
        assert repositories.index_cleanups.list_pending() == []
