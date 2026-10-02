"""
One way for the app to report a failure, and one shape a client reads it in.

Routes used to catch their own exceptions and hand-write a response, which meant
the same failure was spelled differently depending on which route noticed it, a
category was sometimes present and sometimes not, and a crash in a route with no
handler came back as an HTML page instead of JSON. All three were ways for a
client to fail differently on the same problem.

So a route no longer describes a failure at all. It raises, or lets something
raise, and this module decides what that failure looks like on the wire. Every
error the app sends has the same three keys:

    {"error": "...", "category": "...", "details": {...}}

``error`` is what a reader is shown, ``category`` is the stable machine-readable
name a client switches on, and ``details`` carries whatever extra that specific
failure has to explain. A failure with nothing extra leaves ``details`` empty
rather than omitting the key, so a client never has to ask whether it is there.

The same builder produces the failure events on the two server-sent event
streams. A stream that has already sent its headers cannot change its status
code, so a failure mid-stream is reported in the body rather than the status —
but it is reported in this shape, so one reader handles both.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException

from repositories.ingestion_jobs import JobConflictError
from services.accounts.chat_settings_service import SettingsError
from services.accounts.secrets_service import SecretsError, SecretsResaveRequiredError
from services.retrieval.base import (
    VectorDimensionError,
    VectorStoreConfigurationError,
    VectorStoreUnavailableError,
)

log = logging.getLogger(__name__)

#: Shown to a reader when a failure has nothing specific to say. Deliberately
#: says nothing about the cause: this is the response for a bug the app has no
#: classification for, and the traceback belongs in the log, not in a reply.
GENERIC_MESSAGE = "Internal server error"

#: The category for a failure the app cannot classify. Named rather than left
#: empty so a client switching on ``category`` always has a value to switch on,
#: and so "we did not know" is distinguishable from a category nobody implemented.
UNCLASSIFIED = "internal"


class AppError(Exception):
    """
    A failure the app understands well enough to name.

    Carrying the status and category on the exception is what lets a route raise
    and stop: it describes what went wrong in its own words and leaves the
    decision about how that is reported to one place. ``details`` is for the
    context a reader genuinely needs to recover — which documents were left
    behind, whether settings must be re-saved, the model catalog a form is built
    from — and not for a restatement of the message.

    Raise a fresh instance per request rather than a module-level one. Raising an
    exception attaches the frame it was raised in to that instance, so a shared
    one accumulates a traceback on every request it serves and holds that memory
    for as long as the process runs. The failures that recur are built by the
    module-level factories below, which return a new one each time.
    """

    status: int = 500
    category: str = UNCLASSIFIED

    def __init__(
        self,
        message: str | None = None,
        *,
        category: str | None = None,
        status: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record the failure's wording, category, status, and any extra context."""
        super().__init__(message or GENERIC_MESSAGE)
        self.message = message or GENERIC_MESSAGE
        if category is not None:
            self.category = category
        if status is not None:
            self.status = status
        self.details: dict[str, Any] = dict(details or {})


class BadRequest(AppError):
    """The request itself cannot be acted on as written."""

    status = 400
    category = "invalid_request"


class NotFound(AppError):
    """The thing the request names is not there."""

    status = 404
    category = "not_found"


class Conflict(AppError):
    """The request is valid but the app is in a state that cannot serve it yet."""

    status = 409
    category = "conflict"


class PayloadTooLarge(AppError):
    """The request body is larger than the app accepts."""

    status = 413
    category = "payload_too_large"


class UnsupportedMediaType(AppError):
    """The request carries something of a kind the app does not read."""

    status = 415
    category = "unsupported_media_type"


class InternalError(AppError):
    """Something failed that the app has no better classification for."""

    status = 500
    category = UNCLASSIFIED


class ServiceUnavailable(AppError):
    """A dependency the request needs is down or misconfigured."""

    status = 503
    category = "service_unavailable"


def refusal(
    message: str,
    category: str = "",
    status: int = 400,
    **details: Any,
) -> AppError:
    """
    Return the error for a refusal a service decided on.

    Chat and the Research Brief decide whether a request can be served before a
    route is involved, and they return that decision rather than raising it —
    the evaluator reads the same refusal as a case outcome, so it is a value and
    not an exception. A route that gets one hands it here, which is what puts it
    through the same shape as everything else without making those services raise
    an error the evaluator would then have to catch.

    The arguments are in the same order as :class:`AppError` — message first,
    then the category and status as keywords — so the one concept is never
    spelled two ways. The category falls back to one derived from the status when
    the refusal did not name one, so a refusal from a service that had nothing to
    add still arrives with a category a client can switch on.
    """
    return AppError(
        message,
        category=category or category_for_status(status),
        status=status,
        details=dict(details),
    )


#: The category used for a refusal that named none. Chosen per status so the
#: fallback still says something: a 400 that reached here was a bad request and a
#: 404 was a missing thing, whatever the service that declined it called it.
_STATUS_CATEGORIES = {
    400: "invalid_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    429: "rate_limited",
    500: UNCLASSIFIED,
    502: "bad_gateway",
    503: "service_unavailable",
    504: "timeout",
}


def category_for_status(status: int) -> str:
    """
    Return the fallback category for a status with nothing more specific.

    Every failure carries a category, so a client switching on one never has to
    handle an absent value. A status with no entry here — one the app never sends
    on purpose — falls back to ``http_error`` below 500 and ``internal`` above
    it, so "the app declined this" and "the app broke" stay distinguishable even
    for a status nobody thought about.
    """
    return _STATUS_CATEGORIES.get(
        status, UNCLASSIFIED if status >= 500 else "http_error"
    )


#: Builds the error for a domain exception. Kept as a callable rather than an
#: ``AppError`` subclass because a domain exception carries its own values — a
#: mismatch knows the two sizes — and those have to reach the payload.
Handler = Callable[[BaseException], AppError]

#: How a known domain exception becomes an error. Ordered: the first entry whose
#: type matches wins, and subclasses are matched before base classes, so a
#: ``VectorDimensionError`` is never caught by the entry for its ``ValueError``
#: parent. Registering an exception here is the whole of what it takes for a
#: failure raised anywhere below a route to be reported consistently — no route
#: has to catch it.
_REGISTRY: list[tuple[type[BaseException], Handler]] = []


def register(exception: type[BaseException], handler: Handler) -> None:
    """
    Declare how one exception type is reported.

    Most registrations are a plain :class:`AppError` subclass, which
    ``_static_handler`` builds for a type that needs nothing from the exception
    itself. The callable form is for the exceptions whose payload comes out of
    them.
    """
    _REGISTRY.append((exception, handler))
    # Most specific first: an exception is matched against its whole MRO, so a
    # type whose MRO is longer is nearer the front of that MRO and has to be
    # tried before its own base classes. Sorting descending by MRO length gives
    # that, so a subclass registration is never shadowed by its parent.
    _REGISTRY.sort(key=lambda entry: len(entry[0].__mro__), reverse=True)


def _static_handler(error_type: type[AppError]) -> Handler:
    """Return a handler that reports the exception as that AppError type."""

    def handle(error: BaseException) -> AppError:
        return error_type(str(error) or error_type.__name__)

    return handle


def _error_for(exception: BaseException) -> AppError:
    """Return the registered error for an exception, or an unclassified one."""
    for exception_type, handler in _REGISTRY:
        if isinstance(exception, exception_type):
            return handler(exception)
    return InternalError()


def _http_exception_error(error: HTTPException) -> AppError:
    """
    Return an AppError carrying a Flask routing failure's own wording.

    The category comes from the status rather than from the exception's name, so
    that a failure Flask rejects a request for and a failure the app rejects the
    same request for are reported under one category. Flask's own name for a 413
    is "request_entity_too_large"; the app's check for an oversized body says
    "payload_too_large", and a client switching on the category cannot tell those
    apart unless they are the same word.
    """
    status = error.code or 500
    return AppError(
        error.description or error.name or GENERIC_MESSAGE,
        category=category_for_status(status),
        status=status,
    )


def error_payload(
    category: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Return the one body shape every failure is sent in.

    Both the HTTP handlers and the server-sent event streams go through this, so
    a client reads the same three keys whether the failure arrived as a status
    code or as an event inside a stream that had already opened.
    """
    return {"error": message, "category": category, "details": dict(details or {})}


def payload_for(app_error: AppError) -> dict[str, Any]:
    """Return the body an error is reported as."""
    return error_payload(app_error.category, app_error.message, app_error.details)


def classify(error: BaseException) -> AppError:
    """Return the error an exception is reported as."""
    if isinstance(error, AppError):
        return error
    if isinstance(error, HTTPException):
        return _http_exception_error(error)
    return _error_for(error)


def _log(app_error: AppError) -> None:
    """
    Record a failure at a level that matches how bad it was.

    A 4xx is a request the app declined on purpose, so it is logged without a
    traceback: nobody needs the stack for a reader who asked for a file that is
    not there. A 5xx is the app failing at something it did not expect, and the
    traceback is the only thing that says where.
    """
    if app_error.status >= 500:
        log.exception(
            "unhandled failure (%s): %s", app_error.category, app_error.message
        )
    else:
        log.info("request declined (%s): %s", app_error.category, app_error.message)


def register_error_handlers(app: Flask) -> None:
    """
    Give the app one exit for every failure.

    Registered for ``Exception`` rather than for each error type so that a route
    which forgets to anticipate a failure still produces this shape instead of
    an HTML traceback. Reached exceptions are logged here too, which is what
    lets routes raise and stop instead of catching to log and return.
    """
    # Without this, debug and testing mode re-raise instead of falling through
    # to the handler below, and the app answers with a debugger page instead of
    # the shape every client is written against.
    app.config["PROPAGATE_EXCEPTIONS"] = False

    @app.errorhandler(Exception)
    def handle(error: Exception):  # noqa: BLE001 - this is the catch-all
        app_error = classify(error)
        _log(app_error)
        return jsonify(payload_for(app_error)), app_error.status


#: Reported when the vector store cannot be reached. Separate from a
#: misconfigured store because the two need different things from whoever reads
#: the error: one to wait, the other to fix configuration.
def _vector_store_unavailable(error: BaseException) -> AppError:
    # Named apart from the generic unavailable category: telemetry classifies
    # failures by this value and the two need different responses from whoever
    # reads them — one to wait, the other to fix configuration.
    return ServiceUnavailable(
        "Vector store is unavailable",
        category="vector_store_unavailable",
    )


def _vector_store_configuration(error: BaseException) -> AppError:
    return InternalError(
        "Vector store configuration is invalid",
        category="vector_store_configuration",
    )


def _vector_dimension_mismatch(error: BaseException) -> AppError:
    # The remediation the exception names is the recovery, so it is passed
    # through rather than replaced with something vaguer.
    return Conflict(str(error), category="vector_dimension_mismatch")


def register_domain_errors() -> None:
    """
    Declare how the domain's own exceptions are reported.

    Called once at startup. Keeping these registrations here rather than in the
    modules that raise them is what stops the mapping from being spread across
    the layers that happen to raise: the vector store's errors are reported the
    same way whether they surface in chat, in a brief, or in a route.
    """
    register(SettingsError, _static_handler(BadRequest))
    register(SecretsError, _settings_unreadable)
    register(SecretsResaveRequiredError, _secrets_resave_required)
    register(JobConflictError, _job_conflict)
    register(VectorStoreUnavailableError, _vector_store_unavailable)
    register(VectorStoreConfigurationError, _vector_store_configuration)
    register(VectorDimensionError, _vector_dimension_mismatch)


def _job_conflict(error: BaseException) -> AppError:
    """Report work already queued for a Document as a conflict."""
    return Conflict(
        "An ingestion job is already active for this document.",
        category="ingestion_job_active",
    )


def _settings_unreadable(error: BaseException) -> AppError:
    """
    Report stored settings the app cannot read.

    The recovery is the same either way — re-save the settings — so the two
    underlying failures read identically to whoever has to act on it, while the
    logs keep them apart.
    """
    log.error("stored settings unreadable: %s", type(error).__name__)
    return InternalError(
        "Stored settings could not be read. Re-save your provider settings, "
        "then try again.",
        category="settings_unreadable",
    )


def _secrets_resave_required(error: BaseException) -> AppError:
    """
    Report a key this build can no longer decrypt.

    ``needs_resave`` is in the details rather than only in the message because
    the interface offers the re-save itself: the reader should not have to
    recognise the wording to be told what to do next.
    """
    log.warning("stale key derivation")
    return BadRequest(
        str(error),
        category="secrets_resave_required",
        details={"needs_resave": True},
    )
