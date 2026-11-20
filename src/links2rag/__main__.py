"""Command line: `links2rag` or `python -m links2rag`.

Syncs the hackerspace asset manager (pdxhackerspace-links) and the equipment
manuals its things link to into the member-tier RAG pipeline: the Postgres
staging store and entity registry, and the Qdrant hs_member collection.

  links2rag                           # serve: scan, sleep, repeat (default)
  links2rag run --mode incremental    # one scan: changed and deleted things
  links2rag run --mode full           # one scan of everything, then reconcile
  links2rag run --mode full --force   # restage and re-embed everything
  links2rag run --mode reindex        # rebuild every point from staging
  links2rag migrate                   # only create or update the schema
  links2rag status                    # recent runs and counts, as JSON
  links2rag healthcheck               # exit 0 if serve is alive

`serve` does a scan, sleeps SCAN_INTERVAL_MINUTES (default 60) and scans
again. Scans are incremental, plus a full one when none has finished in the
last FULL_SCAN_INTERVAL_HOURS (default 24), so the first scan is full. With
FORCE_REINDEX=true the first scan also restages and re-embeds everything.
Every command except healthcheck first brings the database schema up to date.
"""

import argparse
import json
import logging
import signal
import sys
import tempfile
import threading
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from types import FrameType

from dotenv import load_dotenv

from links2rag import __version__, pipeline
from links2rag.config import ConfigError, Settings
from links2rag.db import migrate
from links2rag.pipeline import RunResult
from links2rag.schedule import Heartbeat, ScanMode, serve

logger = logging.getLogger("links2rag")

HEARTBEAT_PATH = Path(tempfile.gettempdir()) / "links2rag.heartbeat"


def _report(result: RunResult, stop: threading.Event) -> int:
    """Log a run's outcome; return its exit code."""
    counts = result.counts
    summary = (
        f"seen={counts.seen} added={counts.added} updated={counts.updated} "
        f"deleted={counts.deleted} quarantined={counts.quarantined} errors={len(counts.errors)}"
    )
    match result.status:
        case "busy":
            logger.warning("Skipped: %s", result.message)
            return 0
        case "valve":
            logger.error("Sync finished without deletions: %s (%s)", result.message, summary)
            return 1
    if stop.is_set():
        logger.info("Stopped early. %s", summary)
    else:
        logger.info("Sync complete. %s", summary)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="links2rag",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "command",
        nargs="?",
        default="serve",
        choices=["serve", "run", "migrate", "status", "healthcheck"],
        help="what to do (default: serve)",
    )
    parser.add_argument(
        "--mode",
        default="full",
        choices=["incremental", "full", "reindex"],
        help="for run: incremental, full (default) or reindex",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="for run and serve: restage and re-embed everything, ignoring content hashes",
    )
    args = parser.parse_args(argv)

    heartbeat = Heartbeat(HEARTBEAT_PATH)
    if args.command == "healthcheck":
        return 0 if heartbeat.is_fresh() else 1

    # A local .env is a convenience for running outside Docker; variables
    # already set in the environment win.
    load_dotenv()
    try:
        settings = Settings.from_env()
    except ConfigError as e:
        parser.exit(2, f"links2rag: {e}\n")

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    # Every command needs the schema, so apply any pending migrations first;
    # when it is already current this is one query.
    try:
        migrate(settings.database_url)
    except Exception:
        logger.exception("Migration failed")
        return 1
    if args.command == "migrate":
        logger.info("Database schema is up to date")
        return 0

    if args.command == "status":
        try:
            report = pipeline.status(settings)
        except Exception:
            logger.exception("Cannot read status")
            return 1
        sys.stdout.write(json.dumps(report, indent=2, default=str) + "\n")
        return 0

    # `docker stop` sends SIGTERM: finish the thing being staged, index what
    # is staged, then stop (serve also stops sleeping).
    stop = threading.Event()

    def on_signal(signum: int, frame: FrameType | None) -> None:
        logger.info("received signal %d, stopping after the current item", signum)
        stop.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    if args.command == "serve":
        logger.info(
            "links2rag %s serving: a scan every %d minutes, a full one at least every %d hours",
            __version__,
            settings.scan_interval_minutes,
            settings.full_scan_interval_hours,
        )
        # FORCE_REINDEX (or --force) applies to the first scan only.
        pending_force = [settings.force_reindex or args.force]

        def scan(mode: ScanMode) -> None:
            force = pending_force[0]
            if force:
                mode = "full"
            logger.info("Starting a %s scan%s", mode, " (forced)" if force else "")
            _report(pipeline.run(settings, mode, stop, heartbeat.beat, force=force), stop)
            pending_force[0] = False

        serve(
            scan=scan,
            last_full=lambda: pipeline.last_full_run(settings),
            interval=timedelta(minutes=settings.scan_interval_minutes),
            full_every=timedelta(hours=settings.full_scan_interval_hours),
            stop=stop,
            beat=heartbeat.beat,
        )
        return 0

    logger.info(
        "links2rag %s starting a %s run%s",
        __version__,
        args.mode,
        " (forced)" if args.force else "",
    )
    try:
        result = pipeline.run(settings, args.mode, stop, force=args.force)
    except Exception:
        # Top level of a one-shot run: log the traceback and fail the run.
        logger.exception("Sync failed")
        return 1
    return _report(result, stop)


if __name__ == "__main__":
    sys.exit(main())
