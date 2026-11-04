"""Regression: HNSW + RLS filter used to return fewer rows than LIMIT.

HNSW walks the graph for ef_search (40) nearest candidates across ALL users, then the RLS filter
drops the other users' rows. A user owning a small slice of a big table got back ~6 of 20.
vector_search turns on pgvector's iterative scan so the index keeps walking until LIMIT is met.
"""

from __future__ import annotations

import numpy as np
import psycopg

from memory_service.db import ADMIN_DSN, user_session
from memory_service.embed import DIM, FakeEmbedder, to_pgvector
from memory_service.search import vector_search

from .conftest import requires_db

pytestmark = requires_db


def _bulk(admin: psycopg.Connection, user: str, n: int, prefix: str, seed: int) -> None:
    v = np.random.default_rng(seed).standard_normal((n, DIM)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    with admin.cursor() as cur, cur.copy("COPY memories (id, user_id, text, embedding) FROM STDIN") as cp:
        for i in range(n):
            cp.write_row((f"{prefix}-{i}", user, f"row {i}", to_pgvector(v[i].tolist())))


def test_small_user_in_big_table_still_gets_full_limit(conn, users):
    a, b = users
    with psycopg.connect(ADMIN_DSN, autocommit=True) as admin:
        _bulk(admin, b, 4000, f"big-{b}", seed=1)  # someone else's 4000 rows
        _bulk(admin, a, 400, f"small-{a}", seed=2)  # our user: 400 rows, ~9% of the table
        admin.execute("ANALYZE memories")
    with user_session(conn, a):
        # The planner only picks HNSW once it expects this user to have many rows (scripts/bench_index.py:
        # 10 users x 2000). Loading that much in a unit test is slow, so take the sort-based plan away
        # instead: same plan shape as the benchmark, on 4400 rows.
        conn.execute("SET LOCAL enable_sort = off")
        plan = "\n".join(
            r[0] for r in conn.execute("EXPLAIN SELECT id FROM memories ORDER BY embedding <=> %s::vector LIMIT 20",
                                       (to_pgvector([1.0 / DIM**0.5] * DIM),)).fetchall()
        )
        ids = vector_search(conn, FakeEmbedder(), "anything", limit=20)
    assert "memories_embedding_hnsw" in plan  # the planner really does pick HNSW here
    assert len(ids) == 20
    assert all(i.startswith(f"small-{a}") for i in ids)
