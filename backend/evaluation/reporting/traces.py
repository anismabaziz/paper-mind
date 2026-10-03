"""
What the recorded traces of real requests say, in the shape a report reads.

An evaluation run measures the case set a reviewer agreed to ask. The traces are
what the application wrote while serving readers, and they are the half nobody
has to take on trust. A report that quotes the first without the second is
describing a script; one that quotes the second without the first cannot say
whether the run was representative. This module is the reader for the second
half, and it reports the same fields the evaluation run reports so the two sit
in one table without being translated.

Nothing here reads text. A trace is read for its durations, its token counts,
its price, and the outcome it ended as, which are the only fields the answer
path writes that a committed report can carry. A file of traces therefore stays
publishable: the question, the answer, the prompt, and the Passage text are
already gone before this module sees one.

Percentiles come from :mod:`evaluation.answers.metrics`, so a trace's p50 and an
evaluation run's p50 are the same definition rather than two that look alike.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from evaluation.answers.metrics import latency_summary

#: The spans a summary reads. A trace of an answer request always has the
#: generation span when a model was called and the retrieval span whenever the
#: question reached the index; a request refused before either has neither, and
#: is counted as an outcome rather than as a latency sample.
GENERATION = "generation"
RETRIEVAL = "retrieval"

#: The three waits a reader has, measured the same way the evaluation run
#: measures them, and the attribute each is recorded under. Total is the
#: request; first token is the reader's patience; retrieval is the part before
#: the model was asked anything, and it is a span's duration rather than an
#: attribute of the request.
MEASURED = (
    ("retrieval_seconds", "retrieval"),
    ("first_token_seconds", "time_to_first_token_seconds"),
    ("total_seconds", "total_latency_seconds"),
)


def load(path: Path | str) -> list[dict[str, Any]]:
    """
    Return the traces one export file holds, in the order they were written.

    A line that is not a trace is skipped rather than raised on: an export file
    is appended to while the application runs, so a torn last line is expected
    and a summary that refused to be produced over it would be useless exactly
    when it is most wanted.
    """
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and "spans" in record:
            records.append(record)
    return records


def _span(record: dict[str, Any], name: str) -> dict[str, Any] | None:
    """Return the first span with this name, or None when the request never ran it."""
    for span in record.get("spans", []):
        if span.get("name") == name:
            return span
    return None


def _rounded(value: float | None) -> float | None:
    """Return a measured duration at the precision a stored report keeps."""
    return None if value is None else round(value, 6)


def _span_seconds(span: dict[str, Any] | None) -> float | None:
    """Return one span's duration in seconds, or None when it did not run."""
    if span is None:
        return None
    duration = span.get("duration_ms")
    return None if duration is None else float(duration) / 1000.0


def _attribute(record: dict[str, Any], name: str) -> Any:
    """Return one attribute of the request itself, or None when it was not recorded."""
    return record.get("attributes", {}).get(name)


#: Where each wait is recorded: a span's duration for the part before the model
#: was asked anything, an attribute of the request itself for the two a reader
#: waits through. One reader per source, so where a number comes from is a table
#: rather than a branch inside the loop that reads them.
_READERS = {
    RETRIEVAL: lambda record: _span_seconds(_span(record, RETRIEVAL)),
    "time_to_first_token_seconds": lambda record: _attribute(
        record, "time_to_first_token_seconds"
    ),
    "total_latency_seconds": lambda record: _attribute(record, "total_latency_seconds"),
}


def _counts(records: Sequence[dict[str, Any]], source: str) -> dict[str, float | None]:
    """Return the percentile summary of one measurement over the traces that took it."""
    read = _READERS[source]
    samples = [
        float(value)
        for record in records
        for value in [read(record)]
        if isinstance(value, (int, float))
    ]
    summary = latency_summary(samples).to_dict()
    # Rounded to the microsecond, which is far below any wait a reader notices,
    # so a stored report does not carry the interpolation's float noise.
    return {
        "samples": summary["samples"],
        "p50": _rounded(summary["p50"]),
        "p95": _rounded(summary["p95"]),
    }


def _cost(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """
    Return what the recorded model calls cost, and the model that charged it.

    The price is not recomputed from the catalog: a trace already carries what
    the call cost at the prices the provider was configured with, and a summary
    that re-derived it would report a different number than the one recorded
    whenever the catalog moved after the request was served. The per-trace mean
    is beside the total because the interesting figure is what a reader pays.
    """
    spans = [_span(record, GENERATION) for record in records]
    calls = [span["attributes"] for span in spans if span is not None]
    spent = [
        float(call["cost_usd"]) for call in calls if call.get("cost_usd") is not None
    ]
    models = {call.get("model") for call in calls}
    models.discard(None)
    return {
        "usd": round(sum(spent), 6),
        "priced_calls": len(spent),
        "per_trace_usd": round(sum(spent) / len(spent), 6) if spent else 0.0,
        "model": ", ".join(sorted(str(model) for model in models)) or None,
    }


def _tokens(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Return the tokens the recorded model calls read and wrote."""
    counted = [span for span in map(lambda r: _span(r, GENERATION), records) if span]
    return {
        "input": sum(
            int(span["attributes"].get("input_tokens") or 0) for span in counted
        ),
        "output": sum(
            int(span["attributes"].get("output_tokens") or 0) for span in counted
        ),
    }


def _throughput(records: Sequence[dict[str, Any]]) -> dict[str, float | None]:
    """
    Return how many answers a reader got per minute of the window they arrived in.

    Throughput is a property of the wall clock the requests were spread over, not
    of how long any one of them took: four requests a minute apart and four
    requests a second apart are the same p50 and four different rates. A reader
    arriving at the rate recorded here is what the number describes, so the
    window is reported next to it rather than assumed.
    """
    moments = [
        (span.get("started_at"), span.get("ended_at"))
        for record in records
        for span in record.get("spans", [])
        if isinstance(span, dict)
    ]
    started = [start for start, _ in moments if isinstance(start, (int, float))]
    ended = [end for _, end in moments if isinstance(end, (int, float))]
    if len(started) < 2:
        return {"traces_per_minute": None, "window_seconds": 0.0}
    window = max(ended + started) - min(started)
    if window <= 0:
        return {"traces_per_minute": None, "window_seconds": 0.0}
    return {
        "traces_per_minute": round(len(records) * 60.0 / window, 3),
        "window_seconds": round(window, 6),
    }


def _outcomes(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    """
    Return what each request ended as, counted under the outcome it really had.

    Every recorded request appears exactly once, including one refused before it
    reached a model: a summary that reported only the requests that produced an
    answer would let a configuration that failed early look faster and cheaper
    than one that answered, which is the whole mistake this count prevents.
    """
    counts: dict[str, int] = {}
    for record in records:
        outcome = _attribute(record, "outcome") or "unrecorded"
        counts[outcome] = counts.get(outcome, 0) + 1
    return counts


def summarize(paths: Iterable[Path | str]) -> dict[str, Any]:
    """
    Return what the traces in these export files measured, as a report reads it.

    One summary covers every file, because a reader's waiting does not restart
    at a file boundary: two exports from the same day are one population, and
    summarizing them separately would report two runs where there was one.
    """
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(load(path))
    return {
        "traces": len(records),
        "latency": {name: _counts(records, source) for name, source in MEASURED},
        "cost": _cost(records),
        "tokens": _tokens(records),
        "throughput": _throughput(records),
        "outcomes": _outcomes(records),
    }
