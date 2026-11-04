"""Isolation is enforced by Postgres, not by our WHERE clauses: every query here has no user filter."""

from __future__ import annotations

import psycopg
import pytest

from memory_service.db import user_session
from memory_service.embed import FakeEmbedder
from memory_service.search import add_memory, search

from .conftest import requires_db

pytestmark = requires_db
emb = FakeEmbedder()


def _count(conn) -> int:
    return conn.execute("SELECT count(*) FROM memories").fetchone()[0]


def test_app_role_is_not_owner_and_cannot_bypass(conn):
    role, bypass, superuser = conn.execute(
        "SELECT current_user, rolbypassrls, rolsuper FROM pg_roles WHERE rolname = current_user"
    ).fetchone()
    owner = conn.execute("SELECT tableowner FROM pg_tables WHERE tablename = 'memories'").fetchone()[0]
    assert role == "app_user" and role != owner
    assert not bypass and not superuser


def test_b_cannot_see_a(conn, users):
    a, b = users
    with user_session(conn, a):
        add_memory(conn, emb, a, "A's secret: allergic to peanuts")
    with user_session(conn, b):
        assert _count(conn) == 0  # plain SELECT, no WHERE: the policy hides A's row
        assert search(conn, emb, "peanuts", mode="fts") == []
        assert search(conn, emb, "peanuts", mode="vector") == []
    with user_session(conn, a):
        assert _count(conn) == 1


def test_b_cannot_update_or_delete_a(conn, users):
    a, b = users
    with user_session(conn, a):
        mid = add_memory(conn, emb, a, "original text")
    with user_session(conn, b):
        assert conn.execute("UPDATE memories SET text = 'hacked' WHERE id = %s", (mid,)).rowcount == 0
        assert conn.execute("DELETE FROM memories WHERE id = %s", (mid,)).rowcount == 0
    with user_session(conn, a):
        assert conn.execute("SELECT text FROM memories WHERE id = %s", (mid,)).fetchone()[0] == "original text"


def test_b_cannot_write_rows_owned_by_a(conn, users):
    a, b = users
    with pytest.raises(psycopg.errors.InsufficientPrivilege):  # WITH CHECK rejects it
        with user_session(conn, b):
            add_memory(conn, emb, a, "planted by B")


def test_no_session_sees_nothing(conn, users):
    a, _ = users
    with user_session(conn, a):
        add_memory(conn, emb, a, "something")
    assert _count(conn) == 0  # forgot user_session -> empty, not everything


def test_setting_does_not_leak_past_transaction(conn, users):
    a, _ = users
    with user_session(conn, a):
        add_memory(conn, emb, a, "something")
    assert conn.execute("SELECT current_setting('app.user_id', true)").fetchone()[0] in (None, "")
