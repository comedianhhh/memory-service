"""Postgres work queue: enqueue (API side) and claim / complete / fail (worker side).

Delivery is at-least-once, like SQS:
- claim() hands a job to one worker and hides it for `visibility_s` (the lease). FOR UPDATE SKIP LOCKED
  lets many workers claim at once without blocking each other or taking the same row.
- A worker that crashes never completes, so the lease runs out and the job is claimable again.
- complete() and fail() only work while the caller still holds the lease (fencing token), so a worker
  that stalled past its lease can't overwrite the work of the worker that took over.
- After max_attempts failures the job goes 'dead' (the dead-letter queue) instead of retrying forever.
Handlers must therefore be idempotent: memory ids are {message_id}-{i} with ON CONFLICT DO NOTHING.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

VISIBILITY_S = 60  # longer than the slowest message takes to embed
BACKOFF_S = (2, 10, 30, 120)  # wait before retry n (then the last value)


@dataclass(frozen=True)
class Job:
    id: int
    user_id: str
    kind: str
    message_id: str
    payload: dict[str, Any]
    attempts: int
    max_attempts: int
    lease: str


def enqueue(conn: psycopg.Connection, user_id: str, kind: str, payload: dict[str, Any]) -> tuple[int, str]:
    """Call inside user_session (RLS: a user can only enqueue for themself). -> (job id, message id)."""
    message_id = uuid.uuid4().hex
    row = conn.execute(
        "INSERT INTO jobs (user_id, kind, message_id, payload) VALUES (%s, %s, %s, %s) RETURNING id",
        (user_id, kind, message_id, Jsonb(payload)),
    ).fetchone()
    return row[0], message_id


def claim(conn: psycopg.Connection, visibility_s: float = VISIBILITY_S) -> Job | None:
    """Take the oldest visible job and lease it. Its own short transaction, committed before the
    work starts, so the lease is visible to every other worker."""
    with conn.transaction():
        row = conn.execute(
            """
            UPDATE jobs SET
                status = 'running',
                attempts = attempts + 1,
                lease = gen_random_uuid(),
                visible_at = now() + make_interval(secs => %s)
            WHERE id = (
                SELECT id FROM jobs
                WHERE status IN ('queued', 'running') AND visible_at <= now()
                ORDER BY visible_at, id
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING id, user_id, kind, message_id, payload, attempts, max_attempts, lease
            """,
            (visibility_s,),
        ).fetchone()
    if row is None:
        return None
    return Job(row[0], str(row[1]), row[2], row[3], row[4], row[5], row[6], str(row[7]))


def complete(conn: psycopg.Connection, job: Job, result: dict[str, Any]) -> bool:
    """Mark done. Call in the same transaction as the job's writes: then 'done' and the data commit
    together. False means the lease was lost; the caller must roll back."""
    cur = conn.execute(
        """UPDATE jobs SET status = 'done', result = %s, finished_at = now(), lease = NULL, last_error = NULL
           WHERE id = %s AND lease = %s AND status = 'running'""",
        (Jsonb(result), job.id, job.lease),
    )
    return cur.rowcount == 1


def fail(conn: psycopg.Connection, job: Job, error: str) -> str:
    """Record a failure: retry later with backoff, or 'dead' after max_attempts. -> new status."""
    dead = job.attempts >= job.max_attempts
    delay = BACKOFF_S[min(job.attempts, len(BACKOFF_S)) - 1]
    with conn.transaction():
        row = conn.execute(
            """UPDATE jobs SET
                   status = %s,
                   last_error = %s,
                   visible_at = now() + make_interval(secs => %s),
                   lease = NULL,
                   finished_at = CASE WHEN %s THEN now() END
               WHERE id = %s AND lease = %s
               RETURNING status""",
            ("dead" if dead else "queued", error[:2000], delay, dead, job.id, job.lease),
        ).fetchone()
    return row[0] if row else "lease_lost"
