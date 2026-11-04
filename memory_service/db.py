"""Connections. The app only ever uses APP_DSN; ADMIN_DSN is for tests and maintenance."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from uuid import UUID

import psycopg
from psycopg_pool import ConnectionPool

# app_user: not the table owner, NOBYPASSRLS -> row level security applies
APP_DSN = os.environ.get("MEMORY_DSN", "postgresql://app_user:app_user@localhost:5433/memory")
# owner / superuser: RLS does not apply. Never used on the request path.
ADMIN_DSN = os.environ.get("MEMORY_ADMIN_DSN", "postgresql://memory:memory@localhost:5433/memory")


def connect(dsn: str = APP_DSN) -> psycopg.Connection:
    # autocommit: nothing runs in an implicit transaction, so user_session's conn.transaction() is always
    # a real BEGIN/COMMIT and never a savepoint inside some earlier transaction that outlives it.
    return psycopg.connect(dsn, autocommit=True)


def _reset(conn: psycopg.Connection) -> None:
    # Runs when a connection goes back to the pool. user_session uses SET LOCAL so nothing should
    # survive anyway; this also wipes a session-level SET somebody adds by mistake later.
    conn.execute("RESET ALL")


def make_pool(dsn: str = APP_DSN, min_size: int = 1, max_size: int = 10) -> ConnectionPool:
    """Opening a Postgres connection costs a TCP + TLS + auth round trip (~several ms) and a backend
    process. A pool keeps a few open and hands them out per request."""
    return ConnectionPool(
        dsn,
        min_size=min_size,
        max_size=max_size,
        kwargs={"autocommit": True},
        reset=_reset,
        open=False,
        name="memory-app",
    )


@contextmanager
def user_session(conn: psycopg.Connection, user_id: str | UUID) -> Iterator[psycopg.Connection]:
    """One transaction scoped to one user.

    set_config(..., is_local => true) is SET LOCAL with a bind parameter: the setting dies when the
    transaction ends, so a pooled connection can't leak user A's id into user B's next request.
    """
    uid = str(UUID(str(user_id)))  # reject anything that isn't a uuid before it reaches SQL
    with conn.transaction():
        conn.execute("SELECT set_config('app.user_id', %s, true)", (uid,))
        yield conn
