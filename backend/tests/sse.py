"""Shared assertions for server-sent event response bodies."""

import json


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse an SSE response into event names and JSON payloads."""
    events = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        name = None
        payload = None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                payload = json.loads(line.removeprefix("data: "))
        events.append((name, payload))
    return events
