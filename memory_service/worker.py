"""Queue worker: python -m memory_service.worker

Loop: claim a job -> do the slow part with no transaction open (chunk + embed) -> one transaction that
inserts the chunks AND marks the job done. If the lease was lost meanwhile, roll the whole thing back.
SIGTERM / Ctrl+C finishes the current job, then exits (a container stop never cuts a job in half;
even if it did, the lease would expire and another worker would redo it).
"""

from __future__ import annotations

import os
import signal
import threading
import time

import psycopg

from . import jobqueue, logs, telemetry
from .db import user_session
from .embed import Embedder, FastEmbedder
from .search import insert_chunks, prepare_message

WORKER_DSN = os.environ.get("WORKER_DSN", "postgresql://worker:worker@localhost:5433/memory")


class LeaseLost(Exception):
    pass


def handle(conn: psycopg.Connection, embedder: Embedder, job: jobqueue.Job) -> dict:
    if job.kind != "message":
        raise ValueError(f"unknown job kind {job.kind!r}")
    p = job.payload
    prepared = prepare_message(embedder, p["text"], p.get("max_words", 80))  # slow; no transaction open
    with user_session(conn, job.user_id):  # RLS: the worker writes as the job's owner
        ids = insert_chunks(conn, job.user_id, job.message_id, prepared, p.get("importance", 0.5))
        result = {"memory_ids": ids}
        if not jobqueue.complete(conn, job, result):
            raise LeaseLost(f"job {job.id}: lease expired and another worker took it")
    return result


def run_once(conn: psycopg.Connection, embedder: Embedder, visibility_s: float = jobqueue.VISIBILITY_S) -> str | None:
    """Process at most one job. -> 'done' / 'queued' (will retry) / 'dead' / 'lease_lost' / None (empty)."""
    job = jobqueue.claim(conn, visibility_s)
    if job is None:
        return None
    # Continue the trace of the request that enqueued the job, so the waterfall shows
    # POST /me/messages/async -> (queue wait) -> job -> chunk / embed / db.insert as one trace.
    with telemetry.resumed(job.payload.get("_trace")):
        with logs.timed("job", job_id=job.id, kind=job.kind, attempt=job.attempts) as log:
            try:
                result = handle(conn, embedder, job)
                log.update(status="done", chunks=len(result["memory_ids"]))
                return "done"
            except LeaseLost:
                log["status"] = "lease_lost"  # the transaction rolled back; the other worker owns it now
                return "lease_lost"
            except Exception as exc:  # noqa: BLE001 - any failure is recorded on the job, not raised
                status = jobqueue.fail(conn, job, f"{type(exc).__name__}: {exc}")
                log.update(status=status, error=type(exc).__name__)
                return status


def main() -> None:
    logs.setup()
    telemetry.setup_worker()
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    embedder = FastEmbedder()
    logs.logger.info("worker", extra={"fields": {"event": "worker", "state": "started"}})
    while not stop.is_set():
        try:
            with psycopg.connect(WORKER_DSN, autocommit=True) as conn:
                while not stop.is_set():
                    if run_once(conn, embedder) is None:
                        stop.wait(0.5)  # queue empty: poll again shortly
        except psycopg.OperationalError as exc:  # DB restart or network blip: reconnect after a pause
            logs.logger.warning("worker", extra={"fields": {"event": "worker", "state": "db_error", "error": str(exc)[:200]}})
            stop.wait(2)
    logs.logger.info("worker", extra={"fields": {"event": "worker", "state": "stopped"}})


if __name__ == "__main__":
    main()
