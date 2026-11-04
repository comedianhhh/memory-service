from __future__ import annotations

import uuid
from collections.abc import Iterator

import psycopg
import pytest

from memory_service.db import ADMIN_DSN, connect


def _db_up() -> bool:
    try:
        psycopg.connect(ADMIN_DSN, connect_timeout=2).close()
        return True
    except psycopg.OperationalError:
        return False


requires_db = pytest.mark.skipif(not _db_up(), reason="Postgres not running: docker compose up -d")


@pytest.fixture
def users() -> Iterator[tuple[str, str]]:
    """Two fresh user ids; their rows are deleted afterwards with the admin role (RLS doesn't apply there)."""
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    yield a, b
    with psycopg.connect(ADMIN_DSN, autocommit=True) as admin:
        admin.execute("DELETE FROM memories WHERE user_id = ANY(%s::uuid[])", ([a, b],))


@pytest.fixture
def conn() -> Iterator[psycopg.Connection]:
    c = connect()  # app_user
    yield c
    c.close()
