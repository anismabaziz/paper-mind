"""Deterministic backend for Playwright browser tests over real services."""

from __future__ import annotations

import atexit
import logging
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

import psycopg
from psycopg import sql
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

BACKEND_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# Tests must not inherit a developer's .env the way the app boot does.
import dotenv  # noqa: E402

dotenv.load_dotenv = lambda *a, **k: False

from settings import (  # noqa: E402
    AuthSettings,
    ChunkingSettings,
    DatabaseSettings,
    EmbeddingSettings,
    FrontendSettings,
    ParsingSettings,
    RerankSettings,
    Settings,
    StorageSettings,
    TelemetrySettings,
    UploadSettings,
    VectorSettings,
)
from tests.fullstack_support import build_application  # noqa: E402

log = logging.getLogger("browser_server")

ADMIN_DATABASE_URL = os.getenv(
    "FULL_STACK_DATABASE_URL",
    "postgresql+psycopg://papermind:papermind@127.0.0.1:55432/papermind",
)
QDRANT_URL = os.getenv("FULL_STACK_QDRANT_URL", "http://127.0.0.1:56333")
PORT = int(os.getenv("BROWSER_BACKEND_PORT", "38201"))
FRONTEND_ORIGIN = os.getenv(
    "BROWSER_FRONTEND_ORIGIN", "http://127.0.0.1:5180,http://localhost:5180"
)

# Runtime settings a spec may mutate to force a stale index. Anything outside
# this allowlist is rejected so a test cannot silently reconfigure the server.
MUTABLE_RUNTIME: dict[str, tuple[str, ...]] = {
    "chunking": ("chunk_size_tokens", "chunk_overlap_tokens"),
    "embedding": ("embedding_model", "revision"),
    "rerank": ("rerank_model", "revision", "enabled"),
    "parsing": ("use_docling",),
}


def _psycopg_url(database_url: str) -> str:
    url = make_url(database_url)
    return url.set(drivername=url.drivername.replace("+psycopg", "")).render_as_string(
        hide_password=False
    )


def _create_database(database_name: str) -> str:
    with psycopg.connect(_psycopg_url(ADMIN_DATABASE_URL), autocommit=True) as conn:
        conn.execute(
            sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name))
        )
    database_url = (
        make_url(ADMIN_DATABASE_URL)
        .set(database=database_name)
        .render_as_string(hide_password=False)
    )
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=BACKEND_DIR,
        env={**os.environ, "DATABASE_URL": database_url},
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"alembic upgrade failed: {result.stderr}")
    return database_url


def _drop_database(database_name: str) -> None:
    try:
        with psycopg.connect(_psycopg_url(ADMIN_DATABASE_URL), autocommit=True) as conn:
            conn.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()",
                (database_name,),
            )
            conn.execute(
                sql.SQL("DROP DATABASE IF EXISTS {}").format(
                    sql.Identifier(database_name)
                )
            )
    except Exception as exc:  # best effort on shutdown
        log.warning("could not drop browser database: %s", exc)


def _coerce(current, value):
    if isinstance(current, bool):
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(current, int):
        return int(value)
    return str(value)


def register_browser_routes(app, parts) -> None:
    """Add isolated test-only routes to the served Flask app."""
    from flask import jsonify, request

    settings = parts["settings"]
    snapshot = {
        group: {field: getattr(getattr(settings, group), field) for field in fields}
        for group, fields in MUTABLE_RUNTIME.items()
    }

    def fail_targets() -> dict:
        repositories = parts["repositories"]
        return {
            "chat": parts["chat"],
            "storage": parts["storage"],
            "vector_store": parts["vector_store"],
            "embeddings": parts["embeddings"],
            "reranker": parts["reranker"],
            "repo_files": repositories.files,
            "repo_settings": repositories.app_settings,
            "repo_conversations": repositories.conversations,
        }

    @app.route("/_test/info", methods=["GET"])
    def test_info():
        return jsonify({"collection": parts["collection_name"]}), 200

    @app.route("/_test/drain", methods=["POST"])
    def test_drain():
        drained = parts["worker_factory"]("browser-test-drain").drain()
        return jsonify({"drained": drained}), 200

    def apply_failure(switch: str):
        data = request.get_json(silent=True) or {}
        target, operation = data.get("target"), data.get("operation")
        targets = fail_targets()
        if target not in targets or not operation:
            return jsonify({"error": "unknown fail target or missing operation"}), 400
        getattr(targets[target], switch)(operation)
        return jsonify({"target": target, "operation": operation}), 200

    @app.route("/_test/fail", methods=["POST"])
    def test_fail():
        return apply_failure("fail")

    @app.route("/_test/unfail", methods=["POST"])
    def test_unfail():
        return apply_failure("unfail")

    @app.route("/_test/unfail-all", methods=["POST"])
    def test_unfail_all():
        for target in fail_targets().values():
            for operation in list(target.failures):
                target.unfail(operation)
        return jsonify({"ok": True}), 200

    @app.route("/_test/runtime", methods=["GET"])
    def test_runtime_get():
        current = {
            group: {field: getattr(getattr(settings, group), field) for field in fields}
            for group, fields in MUTABLE_RUNTIME.items()
        }
        return jsonify({"runtime": current}), 200

    @app.route("/_test/runtime", methods=["POST"])
    def test_runtime_set():
        data = request.get_json(silent=True) or {}
        applied: dict[str, dict[str, object]] = {}
        for group, changes in data.items():
            if group not in MUTABLE_RUNTIME or not isinstance(changes, dict):
                return jsonify({"error": f"unknown runtime group: {group}"}), 400
            for field, value in changes.items():
                if field not in MUTABLE_RUNTIME[group]:
                    return jsonify(
                        {"error": f"unknown runtime field: {group}.{field}"}
                    ), 400
                section = getattr(settings, group)
                setattr(section, field, _coerce(getattr(section, field), value))
                applied.setdefault(group, {})[field] = getattr(section, field)
        return jsonify({"applied": applied}), 200

    @app.route("/_test/runtime/reset", methods=["POST"])
    def test_runtime_reset():
        for group, fields in snapshot.items():
            section = getattr(settings, group)
            for field, value in fields.items():
                setattr(section, field, value)
        return jsonify({"ok": True}), 200


def main() -> int:
    """Start the browser backend, drain ingestion, and clean up on exit."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if os.getenv("PAPERMIND_BROWSER_TESTS") != "1":
        print("refusing to start: set PAPERMIND_BROWSER_TESTS=1", file=sys.stderr)
        return 2

    database_name = os.getenv(
        "BROWSER_DATABASE", f"papermind_browser_{uuid.uuid4().hex}"
    )
    collection_name = os.getenv("BROWSER_COLLECTION", f"browser-{uuid.uuid4().hex}")
    tmp_dir = tempfile.mkdtemp(prefix="papermind-browser-")
    log.info("creating browser database %s", database_name)
    database_url = _create_database(database_name)
    engine = create_engine(database_url, pool_pre_ping=True)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    app_settings = Settings(
        database=DatabaseSettings(database_url=database_url),
        storage=StorageSettings(storage_dir=pathlib.Path(tmp_dir) / "storage"),
        vector=VectorSettings(qdrant_url=QDRANT_URL, index_name=collection_name),
        embedding=EmbeddingSettings(embedding_model="deterministic"),
        chunking=ChunkingSettings(chunk_size_tokens=512, chunk_overlap_tokens=50),
        rerank=RerankSettings(rerank_model="deterministic", enabled=True),
        parsing=ParsingSettings(use_docling="false"),
        auth=AuthSettings(app_secret="browser-test-secret"),
        upload=UploadSettings(),
        telemetry=TelemetrySettings(
            local_export_path=pathlib.Path(tmp_dir) / "traces" / "answer-traces.jsonl"
        ),
        frontend=FrontendSettings(frontend_origin=FRONTEND_ORIGIN),
    )
    harness = build_application(
        app_settings,
        session_factory,
        register_test_routes=register_browser_routes,
        port=PORT,
    )

    stop = threading.Event()

    def auto_drain() -> None:
        worker = harness.build_worker("browser-auto-drain")
        while not stop.wait(0.4):
            try:
                worker.drain()
            except Exception as exc:
                log.warning("auto-drain failed: %s", type(exc).__name__)

    drain_thread = threading.Thread(target=auto_drain, daemon=True)
    drain_thread.start()

    def shutdown(*_args) -> None:
        stop.set()
        try:
            harness.close()
        finally:
            engine.dispose()
            _drop_database(database_name)

    atexit.register(shutdown)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: (shutdown(), sys.exit(0)))

    print(f"BROWSER_BACKEND_READY url={harness.client.base_url}", flush=True)
    log.info("browser backend on %s", harness.client.base_url)
    # Keep the process alive; Playwright waits for the ready line above.
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
