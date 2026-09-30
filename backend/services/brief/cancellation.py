"""
Stopping a brief that is already running.

A reader who walks away closes the stream, and the run is abandoned: whatever
the brief had collected goes with the connection. That is the right outcome for
someone who left the page, and the wrong one for someone who pressed stop, who
is still there and still wants what the brief found before it stopped.

So a brief can be cancelled while its stream is open. The reader posts the
brief's id, the loop notices between turns and finishes normally, and the reader
receives a terminal event carrying the evidence collected so far with an
explicit incomplete status. Nothing about the brief is thrown away and nothing is
presented as finished.

The registry is process-local, which is what this app is: one reader, one Flask
process, one brief per stream. A brief whose stream has already closed is simply
not in the registry, and a cancel for it is refused rather than silently
succeeding.
"""

from __future__ import annotations

import threading

#: brief id -> the event its loop checks between turns.
_runs: dict[str, threading.Event] = {}
_guard = threading.Lock()


def register(brief_id: str) -> threading.Event:
    """Open a cancellation slot for one brief and return its event."""
    cancel = threading.Event()
    with _guard:
        _runs[brief_id] = cancel
    return cancel


def cancel(brief_id: str) -> bool:
    """
    Ask one running brief to stop, and report whether it was running.

    The event is set rather than the loop interrupted: the loop is the only
    thing that can stop at a point where its evidence is still consistent, and
    a brief stopped anywhere else would leave a half-written turn behind.
    """
    with _guard:
        cancel_event = _runs.get(brief_id)
    if cancel_event is None:
        return False
    cancel_event.set()
    return True


def release(brief_id: str) -> None:
    """Close one brief's slot, whether it finished or was cancelled."""
    with _guard:
        _runs.pop(brief_id, None)


def reset() -> None:
    """Forget every open brief, so one test's runs cannot reach another's."""
    with _guard:
        _runs.clear()
