#!/usr/bin/env python
"""
Seed the local workspace with licensed sample documents.

Uploads the bounded CC0 demo set through the public HTTP routes, waits for
the worker to index each document, then stores one safe sample conversation:
an abstained turn on a nonsense query. Abstention happens before generation,
so seeding never needs a hosted model key.

Idempotent: documents whose ``original_filename`` already exists are skipped,
and an interrupted run resumes without deleting user data.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

BACKEND_DIR = pathlib.Path(__file__).resolve().parent.parent
SAMPLE_DIR = BACKEND_DIR / "evaluation" / "sample_docs"

# Licensed files this seeder may copy. Both are authored in-repo under CC0;
# see evaluation/sample_docs/SOURCE.md. The set stays at two so a fresh
# workspace indexes in minutes on CPU.
SEED_FILES = [
    "papermind-rag-primer.pdf",
    "papermind-team-notes.pdf",
]

SEED_LICENSES = {
    "papermind-rag-primer.pdf": "CC0-1.0",
    "papermind-eval-methods.pdf": "CC0-1.0",
    "papermind-team-notes.pdf": "CC0-1.0",
    "bruening-2018-wearable-jump-monitor-figure-skating.pdf": "CC-BY-4.0",
}


def choose_seed_docs() -> list[pathlib.Path]:
    """Return the bounded licensed PDFs used for the local demo."""
    return [SAMPLE_DIR / name for name in SEED_FILES]


def find_existing(files: list[dict], original_filename: str) -> dict | None:
    """Return the /files entry matching an upload name, if present."""
    for entry in files:
        if entry.get("original_filename") == original_filename:
            return entry
    return None


def build_abstention_query() -> str:
    """Return a query no document can answer, so the app abstains keylessly."""
    return "xqzzy plugh nonexistent term 987654321 what color is this imaginary word?"


def _request(base: str, method: str, path: str, body=None, files=None):
    import http.client
    import urllib.parse

    parsed = urllib.parse.urlsplit(base)
    host, port = parsed.hostname or "127.0.0.1", parsed.port or 3000
    connection = http.client.HTTPConnection(host, port, timeout=60)
    try:
        if files is not None:
            boundary = "papermind-seed-boundary"
            chunks: list[bytes] = []
            for field, (filename, content, mime) in files.items():
                chunks.append(f"--{boundary}\r\n".encode())
                chunks.append(
                    f'Content-Disposition: form-data; name="{field}"; '
                    f'filename="{filename}"\r\n'.encode()
                )
                chunks.append(f"Content-Type: {mime}\r\n\r\n".encode())
                chunks.append(content)
                chunks.append(b"\r\n")
            chunks.append(f"--{boundary}--\r\n".encode())
            payload = b"".join(chunks)
            connection.request(
                method,
                path,
                body=payload,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            )
        elif body is not None:
            payload = json.dumps(body).encode()
            connection.request(
                method,
                path,
                body=payload,
                headers={"Content-Type": "application/json"},
            )
        else:
            connection.request(method, path)
        response = connection.getresponse()
        raw = response.read().decode(errors="replace")
        try:
            data = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            data = {"_raw": raw}
        return response.status, data
    finally:
        connection.close()


def _get(base: str, path: str):
    status, data = _request(base, "GET", path)
    if status != 200:
        raise RuntimeError(f"GET {path} -> {status}: {data}")
    return data


def _wait_ready(base: str, filename: str, timeout_s: float = 600.0) -> None:
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        status, data = _request(
            base, "POST", "/file/is-processed", {"filename": filename}
        )
        if status == 200 and data is not None:
            last = (
                f"{data.get('is_processed')}/{data.get('ingestion', {}).get('state')}"
            )
            if data.get("is_processed") is True:
                return
            if data.get("ingestion", {}).get("state") == "failed":
                raise RuntimeError(f"indexing failed for {filename}: {data}")
        else:
            last = f"http-{status}"
        time.sleep(3)
    raise RuntimeError(f"index never became ready for {filename}: {last}")


def main() -> int:
    """Upload missing samples, wait for indexing, store one abstained turn."""
    parser = argparse.ArgumentParser(description="Seed licensed sample documents")
    parser.add_argument(
        "--base-url", default=os.getenv("PAPERMIND_API_URL", "http://127.0.0.1:3000")
    )
    parser.add_argument("--timeout-s", type=float, default=600.0)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    try:
        files = _get(base, "/files").get("files", [])
    except Exception as exc:
        print(f"Seed failed: backend not reachable at {base}: {exc}", file=sys.stderr)
        print(
            "Fix: run ./papermind.sh up first, then retry with --seed.", file=sys.stderr
        )
        return 1

    seeded = 0
    for doc in choose_seed_docs():
        if not doc.exists():
            print(f"  skip     {doc.name}: sample PDF missing", file=sys.stderr)
            continue
        if find_existing(files, doc.name):
            print(f"  ok       {doc.name}: already present, skipping")
            continue
        print(f"  upload   {doc.name} ({SEED_LICENSES[doc.name]})")
        content = doc.read_bytes()
        status, data = _request(
            base,
            "POST",
            "/upload",
            files={"file": (doc.name, content, "application/pdf")},
        )
        if status != 200:
            print(f"  FAILED   {doc.name}: upload -> {status}: {data}", file=sys.stderr)
            return 1
        filename = data["file"]["name"]
        status, _ = _request(base, "POST", "/process-file", {"filename": filename})
        if status not in (200, 201, 202):
            print(f"  FAILED   {doc.name}: process-file -> {status}", file=sys.stderr)
            return 1
        _wait_ready(base, filename, timeout_s=args.timeout_s)
        print(f"  ready    {doc.name} -> {filename}")
        seeded += 1
        files = _get(base, "/files").get("files", [])

    # One safe sample exchange: a nonsense question abstains without spending
    # on generation, leaving a stored conversation a reviewer can open. Chat
    # binds the saved provider before retrieval runs, so a workspace with no
    # key yet keeps its documents and skips this step instead of failing.
    primer = find_existing(files, SEED_FILES[0])
    if primer is not None:
        query = build_abstention_query()
        try:
            with urllib.request.urlopen(
                urllib.request.Request(
                    f"{base}/response",
                    data=json.dumps(
                        {"filename": primer["name"], "query": query}
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                ),
                timeout=120,
            ) as response:
                response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 400:
                print(
                    "  sample   skipped: no provider key saved yet (open Settings, then rerun)"
                )
            else:
                print(f"  FAILED   sample exchange -> HTTP {exc.code}", file=sys.stderr)
                return 1
        else:
            print(f"  sample   turn stored on {SEED_FILES[0]}")

    print(f"Seed complete: {seeded} new document(s), others already present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
