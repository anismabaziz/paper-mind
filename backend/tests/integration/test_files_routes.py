"""Integration tests for document routes over fakes.

Needs ``pytest tests/integration``. Real Flask app, real temp storage, fake
repositories and vector service.
"""

import io

import pytest

from conftest import (
    FakeConversations,
    FakeFiles,
    FakeIngestionJobs,
    FakeVectorService,
    JobConflictError,
    make_client,
    make_file_record,
    make_pdf_bytes,
    make_ready_record,
    make_services,
)

pytestmark = pytest.mark.integration


def _upload(client, content, filename="paper.pdf", content_type="application/pdf"):
    return client.post(
        "/upload",
        data={"file": (io.BytesIO(content), filename, content_type)},
        content_type="multipart/form-data",
    )


class TestHealth:
    def test_health(self, isettings):
        client = make_client(make_services(isettings), isettings)
        assert client.get("/health").get_json() == {"response": "OK"}


class TestUpload:
    def test_success(self, isettings):
        services = make_services(isettings)
        response = _upload(make_client(services, isettings), make_pdf_bytes())
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["file"]["title"] == "Test Paper"
        assert payload["job"]["state"] == "queued"

    def test_no_file(self, isettings):
        response = make_client(make_services(isettings), isettings).post("/upload")
        assert response.status_code == 400
        assert response.get_json()["category"] == "file_missing"

    def test_wrong_extension(self, isettings):
        response = _upload(
            make_client(make_services(isettings), isettings),
            b"data",
            "doc.txt",
            "text/plain",
        )
        assert response.status_code == 415

    def test_empty_file(self, isettings):
        response = _upload(make_client(make_services(isettings), isettings), b"")
        assert response.status_code == 400
        assert response.get_json()["category"] == "file_empty"

    def test_not_a_pdf(self, isettings):
        response = _upload(
            make_client(make_services(isettings), isettings), b"hello world"
        )
        assert response.status_code == 415

    def test_oversize_content_length(self, isettings):
        isettings.upload.max_upload_bytes = 10
        client = make_client(make_services(isettings), isettings)
        response = client.post(
            "/upload",
            data={"file": (io.BytesIO(b"x" * 100), "paper.pdf", "application/pdf")},
            content_type="multipart/form-data",
            headers={"Content-Length": "1000"},
        )
        assert response.status_code == 413

    def test_db_failure_cleans_orphan(self, isettings):
        class BadFiles(FakeFiles):
            def create_file_with_job(self, *args, **kwargs):
                raise RuntimeError("db down")

        services = make_services(isettings, files=BadFiles())
        response = _upload(make_client(services, isettings), make_pdf_bytes())
        assert response.status_code == 500
        assert services.storage.list() == []


class TestDocumentReads:
    def _client_with_doc(self, isettings, record=None):
        record = record or make_ready_record(isettings)
        services = make_services(isettings, files=FakeFiles([record]))
        services.storage.save(record["filename"], make_pdf_bytes())
        return make_client(services, isettings), services

    def test_meta(self, isettings):
        client, _ = self._client_with_doc(isettings)
        response = client.get("/files/doc1.pdf/meta")
        assert response.status_code == 200
        assert response.get_json()["pageCount"] == 1

    def test_meta_missing_bytes_is_404(self, isettings):
        services = make_services(
            isettings, files=FakeFiles([make_ready_record(isettings)])
        )
        response = make_client(services, isettings).get("/files/doc1.pdf/meta")
        assert response.status_code == 404

    def test_meta_traversal(self, isettings):
        client, _ = self._client_with_doc(isettings)
        assert client.get("/files/../x.pdf/meta").status_code == 400

    def test_download(self, isettings):
        client, _ = self._client_with_doc(isettings)
        response = client.get("/storage/doc1.pdf")
        assert response.status_code == 200
        assert response.data.startswith(b"%PDF")

    def test_is_processed(self, isettings):
        client, _ = self._client_with_doc(isettings)
        response = client.post("/file/is-processed", json={"filename": "doc1.pdf"})
        assert response.status_code == 200
        assert response.get_json()["is_processed"] is True

    def test_is_processed_unknown(self, isettings):
        client = make_client(make_services(isettings), isettings)
        assert (
            client.post("/file/is-processed", json={"filename": "no.pdf"}).status_code
            == 404
        )

    def test_opened(self, isettings):
        client, _ = self._client_with_doc(isettings)
        response = client.post("/file/opened", json={"filename": "doc1.pdf"})
        assert response.status_code == 200
        assert "last_opened_at" in response.get_json()

    def test_opened_unknown(self, isettings):
        client = make_client(make_services(isettings), isettings)
        assert (
            client.post("/file/opened", json={"filename": "no.pdf"}).status_code == 404
        )

    def test_files_list_backfills_hex_title(self, isettings):
        record = make_ready_record(isettings, title="d" * 32)
        services = make_services(isettings, files=FakeFiles([record]))
        services.storage.save(record["filename"], make_pdf_bytes(title="Derived Name"))
        response = make_client(services, isettings).get("/files")
        assert response.status_code == 200
        (entry,) = response.get_json()["files"]
        assert entry["title"] == "Derived Name"
        assert services.repositories.files.titles[record["filename"]] == "Derived Name"

    def test_files_list_skips_unsafe_names(self, isettings):
        record = make_ready_record(isettings, filename="../evil.pdf")
        services = make_services(isettings, files=FakeFiles([record]))
        response = make_client(services, isettings).get("/files")
        assert response.get_json()["files"] == []


class TestIngestionJobs:
    def test_get_job(self, isettings):
        job = {"id": "job-1", "state": "running"}
        services = make_services(
            isettings,
            files=FakeFiles([make_ready_record(isettings)]),
            jobs=FakeIngestionJobs(latest=job),
        )
        client = make_client(services, isettings)
        assert client.get("/ingestion-jobs/doc1.pdf").get_json()["job"] == job

    def test_get_job_none(self, isettings):
        services = make_services(
            isettings, files=FakeFiles([make_ready_record(isettings)])
        )
        response = make_client(services, isettings).get("/ingestion-jobs/doc1.pdf")
        assert response.status_code == 404
        assert response.get_json()["category"] == "ingestion_job_not_found"

    def test_cancel(self, isettings):
        job = {"id": "job-1", "state": "cancelling"}
        services = make_services(
            isettings,
            files=FakeFiles([make_ready_record(isettings)]),
            jobs=FakeIngestionJobs(latest=job),
        )
        response = make_client(services, isettings).post(
            "/ingestion-jobs/doc1.pdf/cancel"
        )
        assert response.status_code == 200
        assert response.get_json()["cancelled"] is True

    def test_cancel_deleting_blocked(self, isettings):
        record = make_ready_record(isettings, deletion_state="deleting")
        services = make_services(
            isettings,
            files=FakeFiles([record]),
            jobs=FakeIngestionJobs(latest={"id": "j", "state": "running"}),
        )
        response = make_client(services, isettings).post(
            "/ingestion-jobs/doc1.pdf/cancel"
        )
        assert response.status_code == 409

    def test_retry_enqueues(self, isettings):
        services = make_services(
            isettings, files=FakeFiles([make_ready_record(isettings)])
        )
        response = make_client(services, isettings).post(
            "/ingestion-jobs/doc1.pdf/retry"
        )
        assert response.status_code == 201

    def test_retry_conflict_carries_job(self, isettings):
        active = {"id": "job-1", "state": "running"}
        services = make_services(
            isettings,
            files=FakeFiles([make_ready_record(isettings)]),
            jobs=FakeIngestionJobs(active=active, conflict=True),
        )
        response = make_client(services, isettings).post(
            "/ingestion-jobs/doc1.pdf/retry"
        )
        assert response.status_code == 409
        assert response.get_json()["category"] == "ingestion_job_active"
        assert response.get_json()["details"]["job"] == active

    def test_reindex_with_active_returns_existing(self, isettings):
        active = {"id": "job-1", "state": "running"}
        services = make_services(
            isettings,
            files=FakeFiles([make_ready_record(isettings)]),
            jobs=FakeIngestionJobs(active=active),
        )
        services.storage.save("doc1.pdf", make_pdf_bytes())
        response = make_client(services, isettings).post("/files/doc1.pdf/reindex")
        assert response.status_code == 202
        assert response.get_json()["job"] == active

    def test_reindex_missing_bytes(self, isettings):
        services = make_services(
            isettings, files=FakeFiles([make_ready_record(isettings)])
        )
        response = make_client(services, isettings).post("/files/doc1.pdf/reindex")
        assert response.status_code == 400
        assert response.get_json()["category"] == "document_not_stored"

    def test_process_file_queues(self, isettings):
        services = make_services(
            isettings, files=FakeFiles([make_ready_record(isettings)])
        )
        services.storage.save("doc1.pdf", make_pdf_bytes())
        response = make_client(services, isettings).post(
            "/process-file", json={"filename": "doc1.pdf"}
        )
        assert response.status_code == 202


class TestRemove:
    def _services(self, isettings, record=None):
        record = record or make_ready_record(isettings)
        services = make_services(
            isettings,
            files=FakeFiles([record]),
            conversations=FakeConversations(),
        )
        services.storage.save(record["filename"], make_pdf_bytes())
        return services

    def test_remove_document(self, isettings):
        services = self._services(isettings)
        response = make_client(services, isettings).delete(
            "/files/remove?path=doc1.pdf"
        )
        assert response.status_code == 200
        assert services.repositories.files.deleted_ids == ["file-1"]
        assert services.repositories.ingestion_jobs.cancelled == ["file-1"]
        assert services.repositories.ingestion_jobs.deleted == ["file-1"]
        assert services.vector_service.deleted_names == ["doc1.pdf"]
        assert ("delete_tree", "conv-1") in services.repositories.conversations.calls

    def test_remove_missing_path_param(self, isettings):
        response = make_client(make_services(isettings), isettings).delete(
            "/files/remove"
        )
        assert response.status_code == 400

    def test_remove_orphan_cleans_vectors_and_bytes(self, isettings):
        services = make_services(isettings)
        services.storage.save("orphan.pdf", b"bytes")
        response = make_client(services, isettings).delete(
            "/files/remove?path=orphan.pdf"
        )
        assert response.status_code == 200
        assert services.vector_service.deleted_names == ["orphan.pdf"]
        assert services.storage.exists("orphan.pdf") is False

    def test_partial_failure_reports_leftovers(self, isettings):
        class BadVectors(FakeVectorService):
            def delete_by_filename(self, filename):
                raise RuntimeError("qdrant down")

        record = make_ready_record(isettings)
        services = make_services(
            isettings,
            files=FakeFiles([record]),
            conversations=FakeConversations(),
            vectors=BadVectors(),
        )
        services.storage.save(record["filename"], make_pdf_bytes())
        response = make_client(services, isettings).delete(
            "/files/remove?path=doc1.pdf"
        )
        assert response.status_code == 409
        payload = response.get_json()
        assert payload["category"] == "document_delete_failed"
        assert "vectors" in payload["details"]["leftovers"]
        assert services.repositories.files.deleted_ids == []

    def test_delete_embeddings(self, isettings):
        services = make_services(isettings)
        response = make_client(services, isettings).post("/delete-embeddings")
        assert response.status_code == 200
        assert services.vector_service.deleted_all == 1
