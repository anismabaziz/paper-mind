"""
Where finished traces go.

Two destinations, and neither is required. :class:`JsonlFileExporter` appends
one JSON object per trace to a local file, which is what an operator reads when
nothing else is running; :class:`OtlpExporter` posts the same records to a
configured collector over HTTP. Both are best effort by contract: a write that
fails is dropped, because a trace that cannot be recorded must not cost a
reader their answer.

There is no third destination on purpose. A trace that is going somewhere the
operator configured is going somewhere they chose, and adding a hosted default
would make observability a decision the operator never made.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from collections.abc import Callable
from typing import Any

from services.telemetry.spans import Trace

log = logging.getLogger(__name__)

#: OTLP/HTTP JSON traces live at this path under a configured base endpoint.
OTLP_TRACES_PATH = "/v1/traces"

#: How long a collector is given before the spans are given up on. Long enough
#: for a local collector, short enough that a hung one is not a hung request.
OTLP_TIMEOUT_SECONDS = 2.0


#: The one HTTP call an exporter makes: a URL, a body, and its headers. It is
#: a callable rather than a client object so a test can hand the exporter a
#: recorder and read exactly what would have gone over the wire.
HttpPost = Callable[..., Any]


class JsonlFileExporter:
    """
    Append finished traces to one local file, one JSON object per line.

    A line-delimited file is readable while the application is running and
    still greppable by trace id afterwards, which is what an operator comparing
    two answers actually does with it.
    """

    def __init__(self, path: str | Path) -> None:
        """Write to this path, creating the directory when it is missing."""
        self._path = Path(path)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        """Return the file traces are appended to."""
        return self._path

    def export(self, trace: Trace) -> None:
        """Append one trace, and say so when the file could not be written."""
        line = json.dumps(trace.to_dict(), default=str)
        try:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(f"{line}\n")
        except OSError as error:
            log.warning("could not write trace to %s: %s", self._path, error)


class OtlpExporter:
    """
    Post finished traces to an OTLP/HTTP collector.

    Only what a collector needs to correlate and read a span is sent: the trace
    id, the span names and timings, and the attributes as key/value pairs. The
    OTLP resource and scope identify the service and the schema the attributes
    are written against, so a collector can group by them.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        post: HttpPost | None = None,
        service_name: str = "papermind",
        timeout_seconds: float = OTLP_TIMEOUT_SECONDS,
    ) -> None:
        """Send to this endpoint, through this HTTP client, as this service."""
        self._endpoint = endpoint.rstrip("/")
        self._post = post
        self._service_name = service_name
        self._timeout = timeout_seconds

    def export(self, trace: Trace) -> None:
        """Post one trace, and say so when the collector could not take it."""
        post: HttpPost = self._post if self._post is not None else self._http_post
        try:
            post(
                f"{self._endpoint}{OTLP_TRACES_PATH}",
                content=json.dumps(self._payload(trace)).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
        except Exception as error:  # noqa: BLE001 - a collector must not fail a request
            log.warning("could not export trace to %s: %s", self._endpoint, error)

    def _payload(self, trace: Trace) -> dict[str, Any]:
        """Return the trace in the OTLP/HTTP JSON shape."""
        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": _attributes({"service.name": self._service_name})
                    },
                    "scopeSpans": [
                        {
                            "scope": {
                                "name": "papermind.telemetry",
                                "version": "1",
                            },
                            "spans": [
                                self._span(trace, span.to_dict())
                                for span in trace.spans
                            ],
                        }
                    ],
                }
            ]
        }

    def _span(self, trace: Trace, span: dict[str, Any]) -> dict[str, Any]:
        """Return one span, carrying the trace's correlation identifiers."""
        attributes = dict(trace.attributes)
        attributes.update(span["attributes"])
        attributes["papermind.trace_id"] = trace.trace_id
        if span["error_category"]:
            attributes["error.type"] = span["error_category"]
        sent = {
            "traceId": trace.trace_id,
            "spanId": span["span_id"],
            "name": f"{trace.name}/{span['name']}",
            "kind": 1,
            "startTimeUnixNano": _unix_nanos(span["started_at"]),
            "endTimeUnixNano": _unix_nanos(span["ended_at"]),
            "attributes": _attributes(attributes),
        }
        if span["error_category"]:
            sent["status"] = {"code": 2, "message": span["error_category"]}
        return sent

    @staticmethod
    def _http_post(url: str, *, content: bytes, headers: dict[str, str]) -> Any:
        """Post one body with the standard library, under a short timeout."""
        import urllib.request

        request = urllib.request.Request(url, data=content, headers=headers)
        with urllib.request.urlopen(request, timeout=OTLP_TIMEOUT_SECONDS) as response:
            return response.read()


def _attributes(values: dict[str, Any]) -> list[dict[str, Any]]:
    """Return attributes in the OTLP key/value shape, typed by their value."""
    return [{"key": str(key), "value": _value(value)} for key, value in values.items()]


def _value(value: Any) -> dict[str, Any]:
    """Return one attribute value in the OTLP shape its Python type maps to."""
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [_value(item) for item in value]}}
    if isinstance(value, dict):
        return {"kvlistValue": {"values": _attributes(value)}}
    return {"stringValue": "" if value is None else str(value)}


def _unix_nanos(timestamp: float | None) -> str:
    """Return a wall-clock second as the nanoseconds a collector reads."""
    return str(int((timestamp or time.time()) * 1_000_000_000))
