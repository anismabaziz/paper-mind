"""
A tracer the tests read back.

The answer path is asked the same question over HTTP whether or not anybody is
watching, so the tests that read a trace do it by pointing the application at a
tracer that keeps its traces in memory. That is the only thing this harness
adds: no assertion reads a private cache or a helper call, only the trace the
application would have exported.
"""

from __future__ import annotations

import pytest

from services.telemetry.redaction import Redactor
from services.telemetry.spans import Sink, Trace, Tracer


class _RecordingExporter:
    """Keep every finished trace, in the order they were finished."""

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.traces: list[Trace] = []

    def export(self, trace: Trace) -> None:
        """Hold one finished trace."""
        self.traces.append(trace)


class RecordedTraces:
    """The traces one test's requests produced."""

    def __init__(self, exporter: _RecordingExporter) -> None:
        """Read the traces this exporter collected."""
        self._exporter = exporter

    @property
    def traces(self) -> list[Trace]:
        """Return the recorded traces, oldest first."""
        return self._exporter.traces

    def last(self) -> Trace:
        """Return the trace of the most recent request."""
        assert self.traces, "no trace was recorded"
        return self.traces[-1]

    def span(self, name: str) -> object:
        """Return the one span with this name on the most recent trace."""
        found = self.last().first(name)
        assert found is not None, f"no {name} span recorded"
        return found


@pytest.fixture
def recording_tracer():
    """
    Keep what the application would have exported, and the record of it.

    The harness redacts exactly as the configured tracer does, so a test reads
    the trace the application would have written down.
    """
    exporter = _RecordingExporter()
    tracer = Tracer([Sink(exporter=exporter, redactor=Redactor())])
    return tracer, RecordedTraces(exporter)


@pytest.fixture
def tracer(recording_tracer):
    """
    Point the composed app at the recording tracer.

    The app fixture builds its answer service from this, so shadowing this
    fixture is the only thing a trace-reading test has to do.
    """
    return recording_tracer[0]


@pytest.fixture
def recorded_traces(recording_tracer):
    """Return the traces the recording tracer collected."""
    return recording_tracer[1]


def trace_for(traces: RecordedTraces) -> Trace:
    """Return the most recent recorded trace."""
    return traces.last()
