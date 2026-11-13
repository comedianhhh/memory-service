"""Async ingestion through the Postgres job queue: the failure modes at-least-once delivery has to survive."""

from __future__ import annotations

import psycopg
import pytest

from memory_service import jobqueue, worker
from memory_service.db import ADMIN_DSN, user_session
from memory_service.embed import FakeEmbedder
from memory_service.search import insert_chunks, prepare_message

from .conftest import requires_db

pytestmark = requires_db

LONG = " ".join(f"word{i}" for i in range(200))  # 200 words -> 3 chunks of <= 80


@pytest.fixture
def wconn():
    # These tests play the worker themselves. A real worker on the same database (the compose `worker`
    # service) would race them for jobs and make them flaky, so refuse loudly instead of guessing.
    busy = _admin(
        "SELECT count(*) FROM pg_stat_activity WHERE usename = 'worker' AND datname = current_database()"
    )[0][0]
    if busy:
        pytest.fail("a queue worker is connected to this database; stop it first: docker compose stop worker")
    c = psycopg.connect(worker.WORKER_DSN, autocommit=True)  # the worker role, not the API role
    yield c
    c.close()


def _admin(sql: str, *params) -> list:
    with psycopg.connect(ADMIN_DSN, autocommit=True) as a:
        cur = a.execute(sql, params)
        return cur.fetchall() if cur.description else []


def _expire_lease(job_id: int) -> None:
    """Fast-forward past the visibility timeout instead of sleeping 60 s."""
    _admin("UPDATE jobs SET visible_at = now() - interval '1 second' WHERE id = %s", job_id)


def _rows_for(message_id: str) -> int:
    return _admin("SELECT count(*) FROM memories WHERE source_id = %s", message_id)[0][0]


@pytest.fixture
def queued(client, account):
    """-> (account, job json) for a 200-word message accepted through the async endpoint.
    Other tests may leave jobs behind, so drain anything older first: these tests claim 'the next job'."""
    _admin("UPDATE jobs SET status = 'done' WHERE status IN ('queued', 'running')")
    a = account()
    r = client.post("/me/messages/async", headers=a["headers"], json={"text": LONG})
    assert r.status_code == 202, r.text
    return a, r.json()


def test_accepted_then_processed_then_searchable(client, queued, wconn):
    a, job = queued
    assert job["status"] == "queued"
    assert client.get(f"/me/jobs/{job['job_id']}", headers=a["headers"]).json()["status"] == "queued"

    assert worker.run_once(wconn, FakeEmbedder()) == "done"

    out = client.get(f"/me/jobs/{job['job_id']}", headers=a["headers"]).json()
    assert out["status"] == "done" and out["attempts"] == 1
    assert out["memory_ids"] == [f"{job['message_id']}-{i}" for i in range(3)]
    hits = client.get("/me/memories/search", params={"q": "word150"}, headers=a["headers"]).json()["results"]
    assert any(h["source_id"] == job["message_id"] for h in hits)


def test_crashed_worker_job_is_redelivered_after_the_lease(queued, wconn):
    _, job = queued
    claimed = jobqueue.claim(wconn)  # worker 1 takes it ... and dies (never completes)
    assert claimed.id == job["job_id"]
    assert worker.run_once(wconn, FakeEmbedder()) is None  # invisible while leased

    _expire_lease(job["job_id"])
    assert worker.run_once(wconn, FakeEmbedder()) == "done"  # worker 2 picks it up
    assert _admin("SELECT attempts FROM jobs WHERE id = %s", job["job_id"])[0][0] == 2


def test_redelivery_after_writes_does_not_duplicate(queued, wconn):
    """Worker 1 wrote the chunks but died before marking the job done; worker 2 redoes the job."""
    _, job = queued
    first = jobqueue.claim(wconn)
    prepared = prepare_message(FakeEmbedder(), LONG)
    with user_session(wconn, first.user_id):
        insert_chunks(wconn, first.user_id, first.message_id, prepared)  # committed, job not done
    assert _rows_for(job["message_id"]) == 3

    _expire_lease(job["job_id"])
    assert worker.run_once(wconn, FakeEmbedder()) == "done"
    assert _rows_for(job["message_id"]) == 3  # ON CONFLICT (id) DO NOTHING: still 3, not 6


def test_stale_worker_cannot_complete_and_its_writes_roll_back(queued, wconn):
    _, job = queued
    slow = jobqueue.claim(wconn)  # worker A stalls past its lease
    _expire_lease(job["job_id"])
    fast = jobqueue.claim(wconn)  # worker B takes over with a new lease
    assert fast.id == slow.id and fast.lease != slow.lease

    with pytest.raises(worker.LeaseLost):
        worker.handle(wconn, FakeEmbedder(), slow)
    assert _rows_for(job["message_id"]) == 0  # A's inserts rolled back with its failed completion

    worker.handle(wconn, FakeEmbedder(), fast)
    assert _rows_for(job["message_id"]) == 3
    assert _admin("SELECT status FROM jobs WHERE id = %s", job["job_id"])[0][0] == "done"


class BrokenEmbedder(FakeEmbedder):
    def embed_passages(self, texts):
        raise RuntimeError("model crashed")


def test_poison_job_retries_with_backoff_then_goes_dead(client, queued, wconn):
    a, job = queued
    _admin("UPDATE jobs SET max_attempts = 2 WHERE id = %s", job["job_id"])

    assert worker.run_once(wconn, BrokenEmbedder()) == "queued"  # attempt 1: back in the queue ...
    assert worker.run_once(wconn, BrokenEmbedder()) is None  # ... but not before its backoff
    _expire_lease(job["job_id"])
    assert worker.run_once(wconn, BrokenEmbedder()) == "dead"  # attempt 2 = max: dead letter

    out = client.get(f"/me/jobs/{job['job_id']}", headers=a["headers"]).json()
    assert out["status"] == "dead" and out["error"] == "RuntimeError: model crashed"
    _expire_lease(job["job_id"])
    assert worker.run_once(wconn, FakeEmbedder()) is None  # dead jobs are never claimed again


def test_someone_elses_job_is_not_found(client, account, queued):
    _, job = queued
    stranger = account()
    assert client.get(f"/me/jobs/{job['job_id']}", headers=stranger["headers"]).status_code == 404


def test_api_role_cannot_claim_jobs(conn):
    """The API's database role may enqueue and read its own jobs, never run the queue."""
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        jobqueue.claim(conn)
