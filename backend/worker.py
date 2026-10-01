"""Run the durable ingestion worker."""

import argparse
import logging
import signal
import time

import settings as settings_module
from composition import Services
from repositories import build_repositories
from services.indexing.manifest import manifest_builder
from services.ingestion.worker import IngestionWorker

log = logging.getLogger(__name__)

POLL_INTERVAL_S = 1.0


def build_worker() -> IngestionWorker:
    """Build a worker from the production dependency graph."""
    app_settings = settings_module.get_settings()
    settings_module.set_settings(app_settings)
    services = Services.from_settings(app_settings)
    return IngestionWorker(
        repositories=services.repositories,
        storage=services.storage,
        parser=services.parser,
        embedding_service=services.embedding_service,
        vector_service=services.vector_service,
        limits=services.settings.ingestion,
        manifest_builder=manifest_builder(services.settings),
    )


def main() -> int:
    """Claim and process ingestion jobs until the process is stopped."""
    parser = argparse.ArgumentParser(description="Run the PaperMind ingestion worker")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process the current queue once, then exit.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=POLL_INTERVAL_S,
        help="Seconds to wait before checking for more work.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    worker = build_worker()
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    log.info("ingestion worker %s started", worker.worker_id)
    while not stopping:
        try:
            processed = worker.drain()
        except Exception:  # noqa: BLE001 - keep polling through transient faults
            log.exception("ingestion worker failed to claim work")
            processed = 0
        if args.once and processed == 0:
            break
        if processed == 0:
            time.sleep(max(args.poll_interval, 0.05))
    log.info("ingestion worker %s stopped", worker.worker_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
