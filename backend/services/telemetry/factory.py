"""
Building the tracer the application serves with.

One place decides where traces go and what each destination may say, so a
deployment configures observability in one place and the answer path only ever
asks a tracer to record. A configuration that exports nowhere produces a tracer
that records nothing, which is why a trace is not something the answer path
has to know how to write.
"""

from __future__ import annotations

from services.telemetry.exporters import JsonlFileExporter, OtlpExporter
from services.telemetry.redaction import Redactor
from services.telemetry.spans import Sink, Tracer
from settings import Settings, TelemetrySettings


def build_tracer(settings: TelemetrySettings) -> Tracer:
    """
    Return the tracer this configuration asks for.

    The local file gets the redactor the configuration chose, so a capture
    window applies to what stays on the machine. A configured collector always
    gets the default redactor: what leaves the process is fingerprints,
    whatever the local window is doing.
    """
    if not settings.enabled:
        return Tracer()
    sinks = [
        Sink(
            exporter=JsonlFileExporter(settings.local_export_path),
            redactor=Redactor(
                capture_text=settings.capture_content,
                capture_window_seconds=settings.capture_window_seconds,
            ),
        )
    ]
    if settings.otlp_endpoint.strip():
        sinks.append(
            Sink(exporter=OtlpExporter(settings.otlp_endpoint), redactor=Redactor())
        )
    return Tracer(sinks)


def tracer_for(settings: Settings) -> Tracer:
    """Return the tracer the running configuration asks for."""
    return build_tracer(settings.telemetry)
