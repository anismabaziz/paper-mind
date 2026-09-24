"""HTTP routes for stored documents and indexing."""

import io
import logging
import os
import threading
import time
import uuid
from typing import TYPE_CHECKING

from flask import Flask, jsonify, request, send_from_directory

from routes.common import (
    deletion_blocked_response,
    file_url,
    is_deleting_record,
    is_safe_filename,
    traversal_check,
    vector_store_error_response,
)
from services.retrieval.base import (
    VectorDimensionError,
    VectorStoreConfigurationError,
    VectorStoreUnavailableError,
)
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
_deletion_blocked_response = deletion_blocked_response


def _timed_call(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    return result, time.time() - start


def register_file_routes(app: Flask, services: "Services") -> None:
    """Register document, upload, processing, and vector routes."""
    files_repository = services.repositories.files
    conversations_repository = services.repositories.conversations
    storage = services.storage
    parser = services.parser
    embedding_service = services.embedding_service
    vector_service = services.vector_service
    storage_dir = services.settings.storage.storage_dir
    max_upload_bytes = services.settings.upload.max_upload_bytes
    allowed_extensions = services.settings.upload.allowed_extensions
    allowed_mime_types = services.settings.upload.allowed_mime_types

    @app.route("/files/<path:filename>/meta", methods=["GET"])
    def get_file_meta(filename):
        guard = traversal_check(storage, filename)
        if guard is not None:
            return guard
        try:
            exists = storage.exists(filename) if hasattr(storage, "exists") else False
            if not exists:
                return jsonify({"error": "File not found"}), 404
        except ValueError:
            log.warning("traversal blocked for %r", filename)
            return jsonify({"error": "Invalid filename"}), 400
        except Exception:
            log.exception("get_file_meta exists check failed for %r", filename)
            return jsonify({"error": "Internal server error"}), 500
        try:
            raw = storage.open(filename)
            if hasattr(raw, "read"):
                raw = raw.read()
            if isinstance(raw, bytearray):
                raw = bytes(raw)
            import pymupdf

            with pymupdf.open("pdf", io.BytesIO(raw)) as document:
                outline = [
                    {"title": item[1], "page": item[2], "level": item[0]}
                    for item in document.get_toc()
                ]
                page_count = len(document)
            return jsonify({"pageCount": page_count, "outline": outline}), 200
        except ValueError:
            log.warning("traversal or invalid pdf for %r", filename)
            return jsonify({"error": "Invalid filename"}), 400
        except Exception:
            log.exception("get_file_meta failed for %r", filename)
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/storage/<path:filename>", methods=["GET"])
    def download_file(filename):
        guard = traversal_check(storage, filename)
        if guard is not None:
            return guard
        try:
            exists = storage.exists(filename) if hasattr(storage, "exists") else False
            if not exists:
                return jsonify({"error": "File not found"}), 404
        except ValueError:
            log.warning("traversal blocked for %r", filename)
            return jsonify({"error": "Invalid filename"}), 400
        except Exception:
            log.exception("download exists check failed for %r", filename)
            return jsonify({"error": "Internal server error"}), 500
        try:
            return send_from_directory(storage_dir, filename)
        except Exception:
            log.exception("download failed for %r", filename)
            return jsonify({"error": "Internal server error"}), 500

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
            return jsonify({"error": "File too large"}), 413
        if "file" not in request.files:
            return jsonify({"error": "No File Provided"}), 400

        uploaded_file = request.files["file"]
        raw_original = (uploaded_file.filename or "").strip()
        original_filename = raw_original or None
        file_ext = os.path.splitext(raw_original)[1].lower() if raw_original else ""
        if file_ext not in allowed_extensions:
            log.warning("upload rejected: invalid extension %r", file_ext)
            return jsonify({"error": "Only PDF files are allowed"}), 400
        mime = (
            (uploaded_file.mimetype or uploaded_file.content_type or "")
            .lower()
            .split(";")[0]
            .strip()
        )
        if mime and mime not in allowed_mime_types:
            log.warning("upload rejected: invalid mime %r", mime)
            return jsonify({"error": "Only PDF files are allowed"}), 400

        unique_filename = f"{uuid.uuid4().hex}{file_ext}"
        try:
            file_content = uploaded_file.read()
        except Exception:
            log.exception("upload read failed for %r", raw_original)
            return jsonify({"error": "Internal server error"}), 500

        if len(file_content) > max_upload_bytes:
            log.warning(
                "upload rejected: file size %s exceeds %s",
                len(file_content),
                max_upload_bytes,
            )
            return jsonify({"error": "File too large"}), 413
        if len(file_content) == 0:
            return jsonify({"error": "Only PDF files are allowed"}), 400
        if not file_content.startswith(b"%PDF"):
            log.warning("upload rejected: missing PDF header for %r", raw_original)
            return jsonify({"error": "Only PDF files are allowed"}), 400

        try:
            title = derive_title(file_content, original_filename)
        except Exception:
            log.exception("derive_title failed for %r", raw_original)
            title = None

        try:
            storage.save(unique_filename, file_content)
            file_record = files_repository.create_file(
                unique_filename,
                title=title,
                original_filename=original_filename,
            )
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
                }
            )
        except Exception:
            try:
                storage.delete(unique_filename)
            except Exception:
                pass
            log.exception("upload failed for %r", raw_original)
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/file/is-processed", methods=["POST"])
    def check_processed():
        try:
            data = request.get_json()
            filename = data.get("filename") if data else None
            if not filename:
                return jsonify({"error": "Filename is required"}), 400
            guard = traversal_check(storage, filename)
            if guard is not None:
                return guard
            file_record = files_repository.get_file(filename)
            if not file_record:
                return jsonify({"error": "File not found"}), 404
            return jsonify({"is_processed": file_record["is_processed"]})
        except Exception:
            log.exception("check_processed failed")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/file/opened", methods=["POST"])
    def mark_opened():
        try:
            data = request.get_json()
            filename = data.get("filename") if data else None
            if not filename:
                return jsonify({"error": "Filename is required"}), 400
            guard = traversal_check(storage, filename)
            if guard is not None:
                return guard
            touched = files_repository.touch_opened(filename)
            if not touched:
                return jsonify({"error": "File not found"}), 404
            return jsonify({"last_opened_at": touched["last_opened_at"]}), 200
        except Exception:
            log.exception("mark_opened failed")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/process-file", methods=["POST"])
    def process_file():
        filename = None
        try:
            data = request.get_json()
            filename = data.get("filename") if data else None
            if not filename:
                return jsonify({"error": "Filename is required"}), 400
            guard = traversal_check(storage, filename)
            if guard is not None:
                return guard

            with _document_lock(filename):
                existing = files_repository.get_file(filename)
                if existing is None:
                    return jsonify({"error": "File not found"}), 404
                if _is_deleting(existing):
                    return _deletion_blocked_response(existing)

                wall_start = time.time()
                if not storage.exists(filename):
                    return jsonify({"error": "Failed to fetch file"}), 400
                # Re-check after the existence check: deletion marks the row
                # first, so a concurrent delete is visible here.
                fresh = files_repository.get_file(filename)
                if fresh is not None and _is_deleting(fresh):
                    return _deletion_blocked_response(fresh)
                file_content = storage.open(filename)
                chunk_objects, parse_elapsed = _timed_call(
                    parser.get_chunk_objects, filename, file_content
                )
                degraded = any(chunk.page_no is None for chunk in chunk_objects)
                if degraded:
                    log.warning(
                        "degraded parse for %s: chunks carry null page numbers",
                        filename,
                    )
                if not chunk_objects:
                    return jsonify({"error": "No text extracted from document"}), 400
                # Final guard before touching the index: a delete that won the
                # lock first leaves the row in deleting state.
                guard_record = files_repository.get_file(filename)
                if guard_record is not None and _is_deleting(guard_record):
                    return _deletion_blocked_response(guard_record)
                try:
                    vector_service.delete_by_filename(filename)
                except Exception as exc:
                    log.warning(
                        "/process-file vector cleanup warning for %s: %s", filename, exc
                    )
                texts = [chunk.text for chunk in chunk_objects]
                try:
                    embeddings, embed_elapsed = _timed_call(
                        embedding_service.embed_texts, texts
                    )
                    _, upsert_elapsed = _timed_call(
                        vector_service.upsert_chunks,
                        embeddings,
                        chunk_objects,
                        filename,
                    )
                except Exception:
                    try:
                        vector_service.delete_by_filename(filename)
                    except Exception as cleanup_exc:
                        log.warning(
                            "/process-file compensation cleanup failed for %s: %s",
                            filename,
                            cleanup_exc,
                        )
                    try:
                        files_repository.set_processed(filename, False)
                    except Exception as state_exc:
                        log.warning(
                            "/process-file compensation state failed for %s: %s",
                            filename,
                            state_exc,
                        )
                    raise

                file_record = files_repository.get_file(filename)
                if file_record is not None and _is_deleting(file_record):
                    try:
                        vector_service.delete_by_filename(filename)
                    except Exception:
                        pass
                    return _deletion_blocked_response(file_record)
                if file_record and not conversations_repository.get_conversation_id(
                    file_record["id"]
                ):
                    conversations_repository.create_conversation(file_record["id"])
                files_repository.set_processed(filename, True)

                log.info(
                    "processed %s: %s chunks | parse %.2fs embed %.2fs "
                    "upsert %.2fs total %.2fs",
                    filename,
                    len(chunk_objects),
                    parse_elapsed,
                    embed_elapsed,
                    upsert_elapsed,
                    time.time() - wall_start,
                )
                if degraded:
                    return jsonify(
                        {
                            "message": "PDF processed",
                            "warning": (
                                "Page numbers could not be detected, so citation "
                                "sources for this document show no page."
                            ),
                        }
                    ), 200
                return jsonify({"message": "PDF processed"}), 200
        except (
            VectorStoreUnavailableError,
            VectorStoreConfigurationError,
        ) as exc:
            log.exception("/process-file vector store failed for %s", filename)
            return vector_store_error_response(exc)
        except VectorDimensionError as exc:
            hint = (
                "Delete embeddings via POST /delete-embeddings and re-ingest "
                "your Documents."
            )
            log.warning(
                "/process-file dimension mismatch for %s: %s", filename or "?", exc
            )
            return jsonify(
                {
                    "error": str(exc),
                    "hint": hint,
                    "category": "vector_dimension_mismatch",
                }
            ), 400
        except Exception:
            log.exception("/process-file failed for %s", filename or "?")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/files", methods=["GET"])
    def get_files():
        try:
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
                        "metadata": {
                            "size": storage_item["size"] if storage_item else 0,
                            "content_type": "application/pdf",
                        },
                    }
                )
            return jsonify({"files": enriched_files}), 200
        except Exception:
            log.exception("get_files failed")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/files/remove", methods=["DELETE"])
    def remove_file():
        try:
            filename = request.args.get("path")
            if not filename:
                return jsonify({"error": "File path required"}), 400
            guard = traversal_check(storage, filename)
            if guard is not None:
                return guard
            with _document_lock(filename):
                file_record = files_repository.get_file(filename)
                if not file_record:
                    # No metadata: still attempt index and file cleanup so a
                    # retry after a partial failure cannot leave silent vectors.
                    leftovers: list[str] = []
                    try:
                        vector_service.delete_by_filename(filename)
                    except Exception:
                        log.exception("vector delete failed for %r", filename)
                        leftovers.append("vectors")
                    try:
                        storage.delete(filename)
                    except Exception:
                        log.exception("storage delete failed for %r", filename)
                        leftovers.append("file")
                    if leftovers:
                        return jsonify(
                            {
                                "error": (
                                    "Deletion incomplete: could not remove "
                                    + ", ".join(leftovers)
                                    + ". Retry deletion."
                                ),
                                "category": "document_delete_failed",
                                "leftovers": leftovers,
                            }
                        ), 500
                    return jsonify(
                        {"message": "File and all its data deleted successfully"}
                    ), 200

                # Durable deleting state first: retries re-enter here and the
                # UI can render deleting vs failed from the listing.
                marked = files_repository.mark_deleting(filename)
                if marked is None:
                    return jsonify({"error": "File not found"}), 404
                file_id = file_record["id"]
                failures: list[str] = []

                # Every point for this document lives under its pdf_name
                # payload filter, covering active and any stale generations.
                try:
                    vector_service.delete_by_filename(filename)
                except Exception:
                    log.exception("vector delete failed for %r", filename)
                    failures.append("vectors")
                try:
                    storage.delete(filename)
                except ValueError:
                    log.warning("traversal delete blocked for %r", filename)
                    try:
                        files_repository.mark_delete_failed(
                            filename,
                            "Could not remove file: invalid filename. Retry deletion.",
                        )
                    except Exception:
                        log.exception(
                            "delete-failed marking failed for %r", filename
                        )
                    return jsonify({"error": "Invalid filename"}), 400
                except Exception:
                    log.exception("storage delete failed for %r", filename)
                    failures.append("file")

                conversation_failed = False
                try:
                    conversation_id = conversations_repository.get_conversation_id(
                        file_id
                    )
                    if conversation_id:
                        conversations_repository.delete_conversation_tree(
                            conversation_id
                        )
                except Exception:
                    log.exception("conversation delete failed for %r", filename)
                    failures.append("conversation")
                    conversation_failed = True

                metadata_failed = False
                if not failures:
                    try:
                        files_repository.delete_file(file_id)
                    except Exception:
                        log.exception("metadata delete failed for %r", filename)
                        failures.append("metadata")
                        metadata_failed = True

                if failures:
                    detail = ", ".join(failures)
                    # Keep the row so a retry has the file id, conversation
                    # id, and filename needed to finish the cleanup.
                    try:
                        files_repository.mark_delete_failed(
                            filename,
                            f"Could not remove {detail}. Retry deletion.",
                        )
                    except Exception:
                        log.exception(
                            "delete-failed marking failed for %r", filename
                        )
                    # A metadata failure after external cleanup succeeded is
                    # still not a success: the row remains for retry.
                    if metadata_failed and not conversation_failed:
                        log.error(
                            "remove_file metadata failed for %r after external cleanup",
                            filename,
                        )
                    return jsonify(
                        {
                            "error": (
                                "Deletion incomplete: could not remove "
                                f"{detail}. Retry deletion."
                            ),
                            "category": "document_delete_failed",
                            "deletion_state": "delete_failed",
                            "leftovers": failures,
                        }
                    ), 500
                with _document_locks_guard:
                    _document_locks.pop(filename, None)
                return jsonify(
                    {"message": "File and all its data deleted successfully"}
                ), 200
        except Exception:
            log.exception("remove_file failed for %r", request.args.get("path"))
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/delete-embeddings", methods=["POST"])
    def delete_embeddings():
        try:
            vector_service.delete_all()
            return jsonify({"message": "Embeddings Deleted"}), 200
        except Exception:
            log.exception("delete_embeddings failed")
            return jsonify({"error": "Internal server error"}), 500
