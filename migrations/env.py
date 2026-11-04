"""Alembic env: plain SQL migrations (no ORM models), run as the table owner."""

from __future__ import annotations

import pathlib
import sys

from alembic import context
from sqlalchemy import create_engine, pool

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from memory_service.db import ADMIN_DSN  # noqa: E402


def _url() -> str:
    # psycopg 3 driver for SQLAlchemy
    return ADMIN_DSN.replace("postgresql://", "postgresql+psycopg://", 1)


def run_offline() -> None:
    context.configure(url=_url(), literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_online() -> None:
    engine = create_engine(_url(), poolclass=pool.NullPool)
    with engine.connect() as conn:
        context.configure(connection=conn)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_offline()
else:
    run_online()
