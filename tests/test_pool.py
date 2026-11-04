"""With a pool, the same physical connection serves user A and then user B. Nothing may carry over."""

from __future__ import annotations

from memory_service.db import make_pool, user_session
from memory_service.embed import FakeEmbedder
from memory_service.search import add_memory

from .conftest import requires_db

pytestmark = requires_db


def test_one_connection_reused_by_two_users_leaks_nothing(users):
    a, b = users
    pool = make_pool(min_size=1, max_size=1)  # forces reuse of the same backend
    pool.open(wait=True)
    try:
        with pool.connection() as conn:
            pid_a = conn.info.backend_pid
            with user_session(conn, a):
                add_memory(conn, FakeEmbedder(), a, "A's memory")
            # a careless session-level SET (not LOCAL) that would otherwise stick to the connection
            conn.execute("SELECT set_config('app.user_id', %s, false)", (a,))
        with pool.connection() as conn:
            assert conn.info.backend_pid == pid_a  # really the same connection
            assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 0  # reset wiped it
            with user_session(conn, b):
                assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 0
    finally:
        pool.close()
