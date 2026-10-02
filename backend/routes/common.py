"""Helpers shared by route groups."""

import logging
import urllib.parse
from typing import Any, NoReturn

from flask import request

from errors import GENERIC_MESSAGE, BadRequest, Conflict, refusal
from services.answering import Refusal
from services.deletion import deletion_block_payload

log = logging.getLogger(__name__)


def file_url(storage: Any, filename: str) -> str:
    """Return an absolute URL for a stored document."""
    return f"{request.host_url.rstrip('/')}{storage.url(filename)}"


def check_filename(storage: Any, filename: str) -> None:
    """
    Refuse a stored filename the storage layer will not accept.

    Raises rather than returning a response, so a route reads this as a guard it
    either passes or stops on. Which routes run it no longer changes how the
    refusal is worded or what category it carries.
    """
    if hasattr(storage, "_path"):
        try:
            storage._path(filename)
        except ValueError:
            log.warning("traversal blocked for %r", filename)
            raise _invalid_filename() from None
        except Exception:
            # The storage layer raised something other than a rejection, which
            # says nothing about whether the name is safe. Fall through to the
            # check below, which does not depend on the storage implementation.
            pass
    if ".." in filename or filename.startswith(("/", "\\")):
        log.warning("traversal blocked for %r", filename)
        raise _invalid_filename()


def _invalid_filename() -> BadRequest:
    """Return the error for a filename the app refuses to act on."""
    return BadRequest("Invalid filename", category="invalid_filename")


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


def raise_refusal(result):
    """
    Return a result that is not a refusal, or raise the refusal it carries.

    Chat and the Research Brief decide whether a request can be served before the
    route is involved, and return that decision rather than raising it — the
    evaluator reads the same refusal as a case outcome, so it is a value and not
    an exception. Handing it here is what puts it through the same shape as every
    other failure without making those services raise something the evaluator
    would then have to catch.
    """
    if isinstance(result, Refusal):
        raise refusal(result.error, result.category, result.status, **result.detail)
    return result


def raise_scope_refusal(detail: dict | None) -> NoReturn:
    """
    Raise the error for a Research Brief scope that cannot be run.

    The scope check names its own status, category, and message — it has to,
    because "Document A is being reindexed" and "select two different Documents"
    are different failures with different recoveries. Everything beside those
    three is context for the reader, such as which Document was named, and rides
    along in the details.
    """
    detail = detail or {}
    known = ("status", "category", "error")
    raise refusal(
        str(detail.get("error", GENERIC_MESSAGE)),
        str(detail.get("category", "")),
        int(detail.get("status", 400)),
        **{key: value for key, value in detail.items() if key not in known},
    )


def deletion_blocked(file_record: dict) -> Conflict:
    """
    Return the error blocking work on a Document that is being deleted.

    Built from the deletion payload rather than worded here, so a Document
    refused for being mid-deletion reads identically whether a route, chat, or
    the evaluator asked. A Document that failed to delete says so in its
    category, because the recovery differs: that one must be retried rather than
    waited out.
    """
    payload = deletion_block_payload(file_record)
    details = {
        key: value for key, value in payload.items() if key not in ("error", "category")
    }
    return Conflict(
        payload["error"],
        category=payload["category"],
        details=details,
    )


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
