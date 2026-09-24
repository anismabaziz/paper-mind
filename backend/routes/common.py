"""Helpers shared by route groups."""

import logging
import urllib.parse
from typing import Any

from flask import jsonify, request

from services.retrieval.base import (
    VectorStoreConfigurationError,
    VectorStoreUnavailableError,
)

log = logging.getLogger(__name__)


def vector_store_error_response(error: Exception):
    """Return the stable HTTP response for a vector-store failure category."""
    if isinstance(error, VectorStoreUnavailableError):
        return jsonify(
            {
                "error": "Vector store is unavailable",
                "category": "vector_store_unavailable",
            }
        ), 503
    if isinstance(error, VectorStoreConfigurationError):
        return jsonify(
            {
                "error": "Vector store configuration is invalid",
                "category": "vector_store_configuration",
            }
        ), 500
    raise TypeError("Unsupported vector-store error")


def file_url(storage: Any, filename: str) -> str:
    """Return an absolute URL for a stored document."""
    return f"{request.host_url.rstrip('/')}{storage.url(filename)}"


def traversal_check(storage: Any, filename: str):
    """Return an invalid-filename response when storage rejects a name."""
    if hasattr(storage, "_path"):
        try:
            storage._path(filename)
        except ValueError:
            log.warning("traversal blocked for %r", filename)
            return jsonify({"error": "Invalid filename"}), 400
        except Exception:
            pass
    elif ".." in filename or filename.startswith(("/", "\\")):
        log.warning("traversal blocked for %r", filename)
        return jsonify({"error": "Invalid filename"}), 400
    return None


def is_safe_filename(storage: Any, filename: str) -> bool:
    """Return whether a stored filename passes the storage traversal guard."""
    if hasattr(storage, "_path"):
        try:
            storage._path(filename)
            return True
        except ValueError:
            return False
        except Exception:
            return ".." not in filename
    return ".." not in filename and not filename.startswith("/")


def is_deleting_record(file_record: dict | None) -> bool:
    """Return whether a document is mid-deletion or failed deletion."""
    if not file_record:
        return False
    return file_record.get("deletion_state") in ("deleting", "delete_failed")


def deletion_blocked_response(file_record: dict):
    """Return the stable 409 response blocking work on a deleting document."""
    if file_record.get("deletion_state") == "delete_failed":
        return jsonify(
            {
                "error": (
                    "This document failed to delete. Retry deletion before "
                    "using it again."
                ),
                "category": "document_delete_failed",
                "deletion_state": "delete_failed",
                "deletion_error": file_record.get("deletion_error"),
            }
        ), 409
    return jsonify(
        {
            "error": "This document is being deleted.",
            "category": "document_deleting",
            "deletion_state": "deleting",
        }
    ), 409


def scrub_api_key_from_text(text: str | None, api_key: str | None) -> str | None:
    """Remove raw, truncated, and URL-encoded API key forms from text."""
    if not text or not api_key:
        return text
    variants = {api_key}
    for quote in (urllib.parse.quote, urllib.parse.quote_plus):
        try:
            variants.add(quote(api_key, safe=""))
        except Exception:
            pass
    for size in (8, 6):
        if len(api_key) >= size:
            for i in range(len(api_key) - size + 1):
                chunk = api_key[i : i + size]
                variants.add(chunk)
                for quote in (urllib.parse.quote, urllib.parse.quote_plus):
                    try:
                        variants.add(quote(chunk, safe=""))
                    except Exception:
                        pass
    if len(api_key) >= 4:
        variants.add(api_key[-4:])
        variants.add(api_key[:4])
    result = text
    for variant in sorted(variants, key=len, reverse=True):
        if variant and len(variant) >= 4 and variant in result:
            result = result.replace(variant, "••••")
    return result
