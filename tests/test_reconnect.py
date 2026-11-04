"""Pooled connections that die while idle (DB restart, failover, idle kill) are replaced on checkout."""

from __future__ import annotations

import psycopg

from memory_service.db import ADMIN_DSN

from .conftest import requires_db

pytestmark = requires_db


def _kill_app_connections() -> int:
    with psycopg.connect(ADMIN_DSN, autocommit=True) as admin:
        rows = admin.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE usename = 'app_user' AND pid <> pg_backend_pid()"
        ).fetchall()
    return len(rows)


def test_request_succeeds_after_server_kills_every_pooled_connection(client, account):
    a = account()
    assert client.get("/me", headers=a["headers"]).status_code == 200
    assert _kill_app_connections() >= 1
    # without check=ConnectionPool.check_connection this is a 500: "terminating connection due to administrator command"
    for _ in range(3):
        assert client.get("/me", headers=a["headers"]).status_code == 200
    assert client.get("/health").json()["status"] == "ok"
