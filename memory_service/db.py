from __future__ import annotations

import os

import psycopg

DSN = os.environ.get("MEMORY_DSN", "postgresql://memory:memory@localhost:5433/memory")


def connect() -> psycopg.Connection:
    return psycopg.connect(DSN)
