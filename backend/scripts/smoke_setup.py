#!/usr/bin/env python
"""
Prove a local workspace answers through citations or abstains honestly.

A clean run asks a nonsense question (no model key needed): the expected
proof is a deterministic ``abstained`` event, or a ``done`` event carrying
sources if the index answered the probe. A workspace with no saved provider
key refuses at the settings boundary instead; that refusal is deterministic
and machine-readable, so it counts as the keyless proof. Anything else is a
failure with recovery guidance.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import urllib.parse


def parse_sse_terminal(sse_text: str) -> tuple[str, dict]:
    """Return the last (event, payload) pair of an SSE stream."""
    name, payload = "", {}
    for block in sse_text.strip().split("\n\n"):
        event_name, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = line.split(":", 1)[1].strip()
        if event_name:
            try:
                payload = json.loads(data) if data else {}
            except json.JSONDecodeError:
                payload = {}
            name = event_name
    return name, payload


def classify_proof(terminal: dict) -> str:
    """Sort a terminal payload into cited-answer, abstention, or failure."""
    if terminal.get("done") is True and terminal.get("sources"):
        return "cited-answer"
    if terminal.get("abstained") is True:
        return "abstention"
    return "failure"


def is_keyless_refusal(status: int, body: str) -> bool:
    """Check whether a /response refusal is the deterministic no-key one."""
    if status != 400:
        return False
    try:
        payload = json.loads(body) if body else {}
    except json.JSONDecodeError:
        return False
    return payload.get("category") == "no_provider_configured"


def _post(base: str, path: str, body: dict) -> tuple[int, str]:
    parsed = urllib.parse.urlsplit(base)
    connection = http.client.HTTPConnection(
        parsed.hostname or "127.0.0.1", parsed.port or 3000, timeout=120
    )
    try:
        connection.request(
            "POST",
            path,
            body=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        return response.status, response.read().decode(errors="replace")
    finally:
        connection.close()


def _get(base: str, path: str) -> tuple[int, dict]:
    parsed = urllib.parse.urlsplit(base)
    connection = http.client.HTTPConnection(
        parsed.hostname or "127.0.0.1", parsed.port or 3000, timeout=30
    )
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        raw = response.read().decode(errors="replace")
        try:
            return response.status, json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return response.status, {"_raw": raw}
    finally:
        connection.close()


def main() -> int:
    """Check health, documents, and proof via abstention or cited answer."""
    parser = argparse.ArgumentParser(description="Smoke-test a local workspace")
    parser.add_argument(
        "--base-url", default=os.getenv("PAPERMIND_API_URL", "http://127.0.0.1:3000")
    )
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    status, health = _get(base, "/health")
    if status != 200:
        print(f"Smoke FAILED: /health -> {status} at {base}", flush=True)
        print(
            "Fix: run ./papermind.sh up; unavailable services mean infra never started.",
            flush=True,
        )
        return 1

    status, listing = _get(base, "/files")
    if status != 200:
        print(f"Smoke FAILED: /files -> {status}: {listing}", flush=True)
        return 1
    files = listing.get("files", [])
    if not files:
        print("Smoke FAILED: workspace holds no documents.", flush=True)
        print(
            "Fix: rerun ./papermind.sh up --seed to index the licensed samples.",
            flush=True,
        )
        return 1

    filename = files[0]["name"]
    nonsense = (
        "xqzzy plugh nonexistent term 987654321 what color is this imaginary word?"
    )
    status, sse = _post(base, "/response", {"filename": filename, "query": nonsense})
    if status != 200:
        # A workspace with no saved provider key refuses chat at the settings
        # boundary by design (the route binds the provider before retrieval
        # runs, so no abstention can stream keylessly). That refusal is
        # deterministic and machine-readable, so it counts as the keyless
        # proof; saving a key unlocks the abstention and cited-answer proofs.
        if is_keyless_refusal(status, sse):
            print(
                "Smoke passed: keyless workspace fails closed (no_provider_configured)."
            )
            print(
                "Save a provider key in Settings, then rerun for the abstention proof."
            )
            return 0
        print(
            f"Smoke FAILED: expected an abstained stream, got HTTP {status}: {sse[:300]}",
            flush=True,
        )
        return 1
    name, terminal = parse_sse_terminal(sse)
    proof = classify_proof(terminal)
    if proof == "abstention":
        print(
            f"Smoke passed: deterministic abstention ({terminal.get('reason')}) on {files[0].get('original_filename')}."
        )
        return 0
    if proof == "cited-answer":
        print(f"Smoke passed: cited answer with {len(terminal['sources'])} source(s).")
        return 0
    print(
        f"Smoke FAILED: terminal event {name!r}: {json.dumps(terminal)[:500]}",
        flush=True,
    )
    print(
        "Fix: check worker logs (backend/data/.local/worker.log) and Qdrant at http://127.0.0.1:6333.",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
