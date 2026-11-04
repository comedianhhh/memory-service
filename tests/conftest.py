from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta

os.environ.setdefault("JWT_SECRET", "test-secret-" + "x" * 32)

import psycopg  # noqa: E402
import pytest  # noqa: E402

from memory_service.db import ADMIN_DSN, connect  # noqa: E402


def _db_up(wait_s: float = float(os.environ.get("DB_WAIT_SECONDS", "20"))) -> bool:
    """Right after `docker compose up` Postgres needs a few seconds before it accepts connections.
    Wait for it instead of silently skipping every DB test."""
    import time

    deadline = time.monotonic() + wait_s
    while True:
        try:
            psycopg.connect(ADMIN_DSN, connect_timeout=2).close()
            return True
        except psycopg.OperationalError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(1)


DB_UP = _db_up()
requires_db = pytest.mark.skipif(not DB_UP, reason="Postgres not running: docker compose up -d")


@pytest.fixture(scope="session", autouse=True)
def _migrated() -> None:
    """Bring a fresh database to the latest schema, so `docker compose up` + `pytest` is enough.
    Idempotent: on an up-to-date database this is a no-op."""
    if not DB_UP:
        return
    import pathlib

    from alembic import command
    from alembic.config import Config

    root = pathlib.Path(__file__).resolve().parents[1]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "migrations"))
    command.upgrade(cfg, "head")


def _cleanup(user_ids: list[str]) -> None:
    if not user_ids:
        return
    with psycopg.connect(ADMIN_DSN, autocommit=True) as admin:
        admin.execute("DELETE FROM memories WHERE user_id = ANY(%s::uuid[])", (user_ids,))
        admin.execute("DELETE FROM users WHERE id = ANY(%s::uuid[])", (user_ids,))  # cascades sessions, tokens


@pytest.fixture
def users() -> Iterator[tuple[str, str]]:
    """Two fresh user ids (no account row); their memories are deleted afterwards as admin."""
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    yield a, b
    _cleanup([a, b])


@pytest.fixture
def conn() -> Iterator[psycopg.Connection]:
    c = connect()  # app_user
    yield c
    c.close()


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kw: float) -> None:
        self.now += timedelta(**kw)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def client(clock):
    from fastapi.testclient import TestClient

    from memory_service.api import app, get_embedder, get_now
    from memory_service.embed import FakeEmbedder

    app.dependency_overrides[get_embedder] = FakeEmbedder
    app.dependency_overrides[get_now] = clock
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def account(client) -> Iterator[Callable[[], dict]]:
    """account() -> {"user_id", "email", "password", "tokens", "headers"}; everything is deleted afterwards."""
    made: list[str] = []

    def make() -> dict:
        email = f"t-{uuid.uuid4().hex[:12]}@example.test"
        password = "correct horse battery"
        r = client.post("/auth/register", json={"email": email, "password": password})
        assert r.status_code == 201, r.text
        user_id = r.json()["user_id"]
        made.append(user_id)
        tokens = client.post("/auth/login", json={"email": email, "password": password}).json()
        return {
            "user_id": user_id,
            "email": email,
            "password": password,
            "tokens": tokens,
            "headers": {"Authorization": f"Bearer {tokens['access_token']}"},
        }

    yield make
    _cleanup(made)
