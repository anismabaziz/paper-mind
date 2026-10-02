"""HTTP routes for stored documents and indexing."""

import io
import logging
import os
import threading
import time
import uuid
from typing import TYPE_CHECKING

from flask import Flask, jsonify, request, send_from_directory

from errors import (
    BadRequest,
    Conflict,
    NotFound,
    PayloadTooLarge,
    UnsupportedMediaType,
    classify,
)
from repositories.ingestion_jobs import JobConflictError
from routes.common import (
    check_filename,
    deletion_blocked,
    file_url,
    is_safe_filename,
)
from services.deletion import is_deleting_record
from services.indexing.state import index_status
from services.titles import backfill_title, derive_title

if TYPE_CHECKING:
    from composition import Services

log = logging.getLogger(__name__)


_document_locks: dict[str, threading.RLock] = {}
_document_locks_guard = threading.RLock()


def _document_lock(filename: str) -> threading.RLock:
    """Return the per-document lock serializing process, chat, and deletion."""
    with _document_locks_guard:
        lock = _document_locks.get(filename)
        if lock is None:
            lock = threading.RLock()
            _document_locks[filename] = lock
        return lock


_is_deleting = is_deleting_record


def _timed_call(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    return result, time.time() - start


def register_file_routes(app: Flask, services: "Services") -> None:
    """Register document, upload, processing, and vector routes."""
    files_repository = services.repositories.files
    conversations_repository = services.repositories.conversations
    ingestion_jobs = services.repositories.ingestion_jobs
    index_cleanups = services.repositories.index_cleanups
    storage = services.storage
    vector_service = services.vector_service
    storage_dir = services.settings.storage.storage_dir
    max_upload_bytes = services.settings.upload.max_upload_bytes
    allowed_extensions = services.settings.upload.allowed_extensions
    allowed_mime_types = services.settings.upload.allowed_mime_types

    def _index_status(file_record: dict) -> dict:
        """Judge a document's index against the running configuration."""
        state = index_status(files_repository, file_record, services.settings)
        return state.to_dict(services.settings)

    def _known_file(filename: str) -> dict:
        """
        Return the Document a request names, refusing if the app has none.

        This is the first check on every Document route, so no route can word
        "there is no such Document" differently from another. It deliberately
        stops here: whether the Document is also on its way out is a separate
        question, asked only by the routes that queue work against it.
        """
        check_filename(storage, filename)
        file_record = files_repository.get_file(filename)
        if not file_record:
            raise _not_found()
        return file_record

    def _workable_file(filename: str) -> dict:
        """
        Return the Document a request names, refusing one that cannot be worked on.

        Used by the routes that queue indexing, reindexing, or cancellation
        against a Document. A Document on its way out is refused before any of
        that is queued, because indexing or reindexing one that is being removed
        is what leaves vectors behind with nothing left to identify them by.
        """
        file_record = _known_file(filename)
        if _is_deleting(file_record):
            raise deletion_blocked(file_record)
        return file_record

    def _require_stored(filename: str) -> None:
        """Refuse a request naming a Document whose file is not in storage."""
        if not storage.exists(filename):
            raise BadRequest(
                "File is missing from storage",
                category="document_not_stored",
            )

    def _require_readable(filename: str) -> None:
        """
        Refuse a request for a Document whose bytes are not in storage.

        A Document can have a metadata row and no file — the upload half of it
        succeeded and the cleanup after a failure did not — and reading one of
        those is a 404 rather than a 500, because there is genuinely nothing to
        read. A storage layer that rejects the name outright is saying the same
        thing about the name as the traversal guard does, so it is reported the
        same way.
        """
        try:
            exists = storage.exists(filename) if hasattr(storage, "exists") else False
        except ValueError:
            log.warning("traversal blocked for %r", filename)
            raise BadRequest(
                "Invalid filename", category="invalid_filename"
            ) from None
        if not exists:
            raise _not_found()

    def _no_ingestion_job() -> NotFound:
        """Return the error for a Document that has never been queued."""
        return NotFound(
            "No ingestion job for this document",
            category="ingestion_job_not_found",
        )

    @app.route("/files/<path:filename>/meta", methods=["GET"])
    def get_file_meta(filename):
        check_filename(storage, filename)
        _require_readable(filename)
        return jsonify(_pdf_outline(_read(storage, filename))), 200

    @app.route("/storage/<path:filename>", methods=["GET"])
    def download_file(filename):
        # Reading a Document is not the same as working on it, so this does not
        # refuse one that is being deleted: someone who had it open while it was
        # removed can still finish reading it.
        check_filename(storage, filename)
        _require_readable(filename)
        return send_from_directory(storage_dir, filename)

    @app.route("/upload", methods=["POST"])
    def upload_file():
        if (
            request.content_length is not None
            and request.content_length > max_upload_bytes
        ):
            log.warning(
                "upload rejected: content_length %s exceeds %s",
                request.content_length,
                max_upload_bytes,
            )
            raise _too_large()
        if "file" not in request.files:
            raise BadRequest("No File Provided", category="file_missing")

        uploaded_file = request.files["file"]
        raw_original = (uploaded_file.filename or "").strip()
        original_filename = raw_original or None
        file_ext = os.path.splitext(raw_original)[1].lower() if raw_original else ""
        if file_ext not in allowed_extensions:
            log.warning("upload rejected: invalid extension %r", file_ext)
            raise _not_a_pdf()
        mime = (
            (uploaded_file.mimetype or uploaded_file.content_type or "")
            .lower()
            .split(";")[0]
            .strip()
        )
        if mime and mime not in allowed_mime_types:
            log.warning("upload rejected: invalid mime %r", mime)
            raise _not_a_pdf()

        unique_filename = f"{uuid.uuid4().hex}{file_ext}"
        file_content = uploaded_file.read()

        if len(file_content) > max_upload_bytes:
            log.warning(
                "upload rejected: file size %s exceeds %s",
                len(file_content),
                max_upload_bytes,
            )
            raise _too_large()
        if len(file_content) == 0:
            raise _empty_upload()
        if not file_content.startswith(b"%PDF"):
            log.warning("upload rejected: missing PDF header for %r", raw_original)
            raise _not_a_pdf()

        title = _title_or_none(file_content, original_filename, raw_original)

        try:
            storage.save(unique_filename, file_content)
            # One transaction creates the document and its first job, so a
            # document can never exist without work queued to index it.
            file_record, job = files_repository.create_file_with_job(
                unique_filename,
                title=title,
                original_filename=original_filename,
            )
        except Exception:
            # The stored file is orphaned unless the save half of this succeeded
            # and the record half did not, so remove it before the failure is
            # reported. Nothing here is worth its own error: the reader is being
            # told the upload failed either way, and a cleanup failure on top of
            # it would only bury that.
            _discard_quietly(storage.delete, unique_filename)
            raise

        return jsonify(
            {
                "message": "File uploaded successfully",
                "file": {
                    "id": file_record["id"],
                    "name": unique_filename,
                    "title": file_record["title"],
                    "original_filename": file_record["original_filename"],
                    "url": file_url(storage, unique_filename),
                },
                "job": job,
            }
        )

    @app.route("/file/is-processed", methods=["POST"])
    def check_processed():
        filename = _filename_from_body()
        file_record = _known_file(filename)
        return jsonify(
            {
                "is_processed": file_record["is_processed"],
                "ingestion": ingestion_jobs.get_latest(filename),
                "index": _index_status(file_record),
            }
        )

    @app.route("/file/opened", methods=["POST"])
    def mark_opened():
        filename = _filename_from_body()
        check_filename(storage, filename)
        touched = files_repository.touch_opened(filename)
        if not touched:
            raise _not_found()
        return jsonify({"last_opened_at": touched["last_opened_at"]}), 200

    @app.route("/ingestion-jobs/<path:filename>", methods=["GET"])
    def get_ingestion_job(filename):
        """Return the newest ingestion job for one document."""
        _known_file(filename)
        job = ingestion_jobs.get_latest(filename)
        if not job:
            raise _no_ingestion_job()
        return jsonify({"job": job}), 200

    @app.route("/ingestion-jobs/<path:filename>/cancel", methods=["POST"])
    def cancel_ingestion_job(filename):
        file_record = _workable_file(filename)
        with _document_lock(filename):
            job = ingestion_jobs.request_cancel(file_record["id"])
        if job is None:
            raise _no_ingestion_job()
        return jsonify(
            {
                "job": job,
                "cancelled": job["state"] in ("cancelling", "cancelled"),
            }
        ), 200

    @app.route("/ingestion-jobs/<path:filename>/retry", methods=["POST"])
    def retry_ingestion_job(filename):
        """Queue a new attempt for a document that is not already active."""
        file_record = _workable_file(filename)
        try:
            job = ingestion_jobs.enqueue(file_record["id"], filename)
        except JobConflictError as conflict:
            # Caught rather than left to the central handler because the failure
            # the reader needs is the job already running, and only this route
            # knows how to look it up. The category and wording still come from
            # the one registration for this exception, so this cannot drift from
            # the way the same failure reads anywhere else.
            app_error = classify(conflict)
            app_error.details["job"] = ingestion_jobs.get_active(file_record["id"])
            raise app_error from None
        return jsonify({"job": job}), 201

    @app.route("/files/<path:filename>/reindex", methods=["POST"])
    def reindex_file(filename):
        """Queue a replacement index for a document whose index went stale."""
        file_record = _workable_file(filename)
        _require_stored(filename)
        active = ingestion_jobs.get_active(file_record["id"])
        if active:
            return jsonify(
                {
                    "message": "A reindex is already running for this document.",
                    "job": active,
                }
            ), 202
        try:
            job = ingestion_jobs.enqueue(file_record["id"], filename)
        except JobConflictError:
            # A reindex already running is not a failure to the caller who asked
            # for one: they get the job to watch and the same answer either way.
            return jsonify(
                {
                    "message": "A reindex is already running for this document.",
                    "job": ingestion_jobs.get_active(file_record["id"]),
                }
            ), 202
        # The active generation stays queryable until the replacement
        # validates, so a failed reindex never leaves the document unusable.
        return jsonify({"job": job, "index": _index_status(file_record)}), 201

    @app.route("/process-file", methods=["POST"])
    def process_file():
        """Queue indexing work for one document; the worker runs the stages."""
        filename = _filename_from_body()
        file_record = _workable_file(filename)
        _require_stored(filename)
        # Upload already queued this document's job. Reuse it so repeated
        # clicks cannot supersede the queued attempt with a new one.
        active = ingestion_jobs.get_active(file_record["id"])
        if active:
            return jsonify(
                {
                    "message": "Ingestion job already queued",
                    "job": active,
                }
            ), 202
        try:
            job = ingestion_jobs.enqueue(file_record["id"], filename)
        except JobConflictError:
            return jsonify(
                {
                    "message": "Ingestion job already active",
                    "job": ingestion_jobs.get_active(file_record["id"]),
                }
            ), 202
        return jsonify(
            {
                "message": "PDF queued for indexing",
                "job": job,
            }
        ), 202

    @app.route("/files", methods=["GET"])
    def get_files():
        db_files = files_repository.list_files()
        storage_items = storage.list()
        storage_map = {item["name"]: item for item in storage_items}
        enriched_files = []
        for db_file in db_files:
            filename = db_file["filename"]
            if not is_safe_filename(storage, filename):
                log.warning("skipping file with invalid name %r", filename)
                continue
            title = backfill_title(
                storage,
                files_repository,
                filename,
                db_file["title"],
                db_file["original_filename"],
            )
            storage_item = storage_map.get(filename)
            enriched_files.append(
                {
                    "id": db_file["id"],
                    "name": filename,
                    "title": title,
                    "original_filename": db_file["original_filename"],
                    "url": file_url(storage, filename),
                    "is_processed": db_file["is_processed"],
                    "last_opened_at": db_file.get("last_opened_at"),
                    "deletion_state": db_file.get("deletion_state", "active"),
                    "deletion_error": db_file.get("deletion_error"),
                    "deletion_attempts": db_file.get("deletion_attempts", 0),
                    "ingestion": ingestion_jobs.get_latest(filename),
                    "index": _index_status(db_file),
                    "metadata": {
                        "size": storage_item["size"] if storage_item else 0,
                        "content_type": "application/pdf",
                    },
                }
            )
        return jsonify({"files": enriched_files}), 200

    @app.route("/files/remove", methods=["DELETE"])
    def remove_file():
        filename = request.args.get("path")
        if not filename:
            raise BadRequest("File path required")
        check_filename(storage, filename)
        with _document_lock(filename):
            file_record = files_repository.get_file(filename)
            if not file_record:
                _remove_orphans(filename)
            else:
                _remove_document(file_record)
            with _document_locks_guard:
                _document_locks.pop(filename, None)
        return jsonify({"message": "File and all its data deleted successfully"}), 200

    @app.route("/delete-embeddings", methods=["POST"])
    def delete_embeddings():
        vector_service.delete_all()
        return jsonify({"message": "Embeddings Deleted"}), 200

    def _remove_orphans(filename: str) -> None:
        """
        Remove what a Document leaves behind when it has no metadata row.

        A retry after a partial failure lands here, and the point is that it
        cannot leave vectors behind silently: whatever could not be removed is
        reported by name so the reader knows the removal did not finish.
        """
        leftovers: list[str] = []
        _collect_failure(
            lambda: vector_service.delete_by_filename(filename),
            "vectors",
            leftovers,
            filename,
        )
        _collect_failure(lambda: storage.delete(filename), "file", leftovers, filename)
        if leftovers:
            raise _deletion_incomplete(leftovers, deletion_state=None)

    def _remove_document(file_record: dict) -> None:
        """
        Remove one Document's storage, vectors, conversations, and metadata.

        The steps are independent and a failure in one does not stop the others,
        because stopping early would leave the remaining data with nothing left
        to identify it by. What each step failed at is accumulated and reported
        together, and the row is kept whenever the removal did not finish so a
        retry still has the file id and conversation id it needs.
        """
        filename = file_record["filename"]
        file_id = file_record["id"]
        # Durable deleting state first: retries re-enter here and the UI can
        # render deleting vs failed from the listing.
        if files_repository.mark_deleting(filename) is None:
            raise _not_found()
        failures: list[str] = []

        # A queued or running job must not index a document that is being
        # removed, so supersede it before external cleanup.
        _collect_failure(
            lambda: ingestion_jobs.cancel_active(file_id),
            "ingestion",
            failures,
            filename,
        )
        # Every point for this document lives under its pdf_name payload filter,
        # covering active and any stale generations.
        _collect_failure(
            lambda: vector_service.delete_by_filename(filename),
            "vectors",
            failures,
            filename,
        )
        try:
            storage.delete(filename)
        except ValueError:
            # The row exists and the name is now rejected, which retrying will
            # not fix, so the row is released rather than left mid-deletion.
            _mark_delete_failed(
                filename,
                "Could not remove file: invalid filename. Retry deletion.",
            )
            raise BadRequest(
                "Invalid filename", category="invalid_filename"
            ) from None
        except Exception:
            log.exception("storage delete failed for %r", filename)
            failures.append("file")

        conversation_id = _conversation_id(file_id)
        if conversation_id:
            _collect_failure(
                lambda: conversations_repository.delete_conversation_tree(
                    conversation_id
                ),
                "conversation",
                failures,
                filename,
            )

        if not failures:
            _collect_failure(
                lambda: _delete_metadata(file_id),
                "metadata",
                failures,
                filename,
            )

        if failures:
            # Keep the row so a retry has the file id, conversation id, and
            # filename needed to finish the cleanup.
            _mark_delete_failed(
                filename,
                f"Could not remove {', '.join(failures)}. Retry deletion.",
            )
            raise _deletion_incomplete(failures, deletion_state="delete_failed")

    def _delete_metadata(file_id) -> None:
        """Remove a Document's job rows, cleanup rows, and its own row."""
        ingestion_jobs.delete_for_file(file_id)
        index_cleanups.delete_for_file(file_id)
        files_repository.delete_file(file_id)

    def _conversation_id(file_id) -> str | None:
        """Return the conversation a Document's points belong to, if any."""
        try:
            return conversations_repository.get_conversation_id(file_id)
        except Exception:
            log.exception("conversation lookup failed for file %s", file_id)
            return None

    def _collect_failure(step, label: str, failures: list[str], filename: str) -> None:
        """
        Run one removal step, recording its label if it fails.

        Every step is attempted regardless of the ones before it, so a failure
        here is recorded rather than raised: raising would abandon the steps that
        could still have succeeded and left the Document in a worse state than
        reporting the partial removal does.
        """
        try:
            step()
        except Exception:
            log.exception("%s cleanup failed for %r", label, filename)
            failures.append(label)

    def _mark_delete_failed(filename: str, reason: str) -> None:
        """Record that a removal did not finish, so the listing can say so."""
        try:
            files_repository.mark_delete_failed(filename, reason)
        except Exception:
            log.exception("delete-failed marking failed for %r", filename)

    def _deletion_incomplete(
        leftovers: list[str],
        deletion_state: str | None,
    ) -> Conflict:
        """Return the error naming what a partial removal could not remove."""
        listed = ", ".join(leftovers)
        details: dict[str, object] = {"leftovers": leftovers}
        if deletion_state is not None:
            details["deletion_state"] = deletion_state
        return Conflict(
            f"Deletion incomplete: could not remove {listed}. Retry deletion.",
            category="document_delete_failed",
            details=details,
        )


def _filename_from_body() -> str:
    """Return the Document a JSON body names, or refuse the request."""
    data = request.get_json(silent=True) or {}
    filename = data.get("filename") if isinstance(data, dict) else None
    if not filename:
        raise BadRequest("Filename is required")
    return str(filename)


def _not_found() -> NotFound:
    """Return the error for a Document the app has no record of."""
    return NotFound("File not found", category="file_not_found")


def _too_large() -> PayloadTooLarge:
    """Return the error for an upload larger than the app accepts."""
    return PayloadTooLarge("File too large")


def _not_a_pdf() -> UnsupportedMediaType:
    """Return the error for an upload the parser cannot read."""
    return UnsupportedMediaType("Only PDF files are allowed")


def _empty_upload() -> BadRequest:
    """
    Return the error for an upload that arrived with nothing in it.

    Deliberately not the same failure as an upload of the wrong kind: an empty
    body is a request that did not carry what it said it would, while a wrong
    extension or a missing PDF header is a kind of file the app cannot read. The
    two read the same to whoever sent them, but only one of them is fixed by
    choosing a different file.
    """
    return BadRequest("The uploaded file is empty", category="file_empty")


def _read(storage, filename: str) -> bytes:
    """Return a Document's bytes for the reader to parse."""
    raw = storage.open(filename)
    if hasattr(raw, "read"):
        raw = raw.read()
    if isinstance(raw, bytearray):
        raw = bytes(raw)
    return raw


def _pdf_outline(raw: bytes) -> dict:
    """Return a Document's page count and table of contents."""
    import pymupdf

    with pymupdf.open("pdf", io.BytesIO(raw)) as document:
        outline = [
            {"title": item[1], "page": item[2], "level": item[0]}
            for item in document.get_toc()
        ]
        page_count = len(document)
    return {"pageCount": page_count, "outline": outline}


def _title_or_none(
    file_content: bytes,
    original_filename: str | None,
    raw_original: str,
) -> str | None:
    """
    Return a Document's derived title, or None when deriving it fails.

    A title is a convenience the reader sees later, so a failure to derive one
    is not a failure to upload: the Document is stored and indexed either way,
    and the title is backfilled from the file when it is next listed. Recorded
    and swallowed rather than raised so a title problem cannot reject an upload.
    """
    try:
        return derive_title(file_content, original_filename)
    except Exception:
        log.exception("derive_title failed for %r", raw_original)
        return None


def _discard_quietly(step, *args) -> None:
    """Run a best-effort cleanup step, ignoring whether it works."""
    try:
        step(*args)
    except Exception:
        log.exception("cleanup after failed upload did not complete")
