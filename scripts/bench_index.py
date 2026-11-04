"""Load ~20k synthetic rows across 10 users, then EXPLAIN ANALYZE the vector and full-text queries
as app_user under RLS. Shows which index each query uses and how many rows actually come back.

    .venv/Scripts/python scripts/bench_index.py            # load, explain, clean up
    .venv/Scripts/python scripts/bench_index.py --keep     # leave the rows for poking at in psql
"""

from __future__ import annotations

import argparse
import pathlib
import random
import re
import sys
import uuid

import numpy as np
import psycopg

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memory_service.db import ADMIN_DSN, connect, user_session  # noqa: E402
from memory_service.embed import DIM, to_pgvector  # noqa: E402

BENCH_NS = uuid.UUID("b0000000-0000-0000-0000-000000000000")
VOCAB = (
    "coffee dentist cat flight rent gym knee trip sister mom boss dashboard kombucha tires birthday "
    "japanese marathon laptop phone desk wedding allergy bouldering volunteer calculus orchid"
).split()


def bench_users(n: int) -> list[str]:
    return [str(uuid.uuid5(BENCH_NS, str(i))) for i in range(n)]


def load(users: list[str], per_user: int, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    rnd = random.Random(seed)
    with psycopg.connect(ADMIN_DSN, autocommit=True) as admin:  # owner/superuser: bulk load, RLS n/a
        admin.execute("DELETE FROM memories WHERE user_id = ANY(%s::uuid[])", (users,))
        with admin.cursor() as cur, cur.copy("COPY memories (id, user_id, text, embedding) FROM STDIN") as cp:
            for u_i, u in enumerate(users):
                vecs = rng.standard_normal((per_user, DIM)).astype(np.float32)
                vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
                for j in range(per_user):
                    words = rnd.choices(VOCAB, k=12)
                    if j % 500 == 0:
                        words.append("zanzibar")  # rare word: 4 rows per user
                    cp.write_row((f"bench-{u_i}-{j}", u, " ".join(words), to_pgvector(vecs[j].tolist())))
        admin.execute("ANALYZE memories")


def explain(conn: psycopg.Connection, sql: str, params: tuple) -> tuple[list[str], float]:
    plan = [r[0] for r in conn.execute("EXPLAIN (ANALYZE, BUFFERS, COSTS OFF) " + sql, params).fetchall()]
    ms = float(re.search(r"Execution Time: ([\d.]+) ms", plan[-1]).group(1))
    return plan, ms


def show(title: str, plan: list[str], ms: float, rows: int) -> None:
    keep = [ln for ln in plan if re.search(r"Scan|Limit|Sort|Filter|Rows Removed|Index Searches", ln)]
    print(f"\n== {title}: {rows} rows, {ms:.2f} ms")
    for ln in keep:
        ln = ln.rstrip()
        print("  " + (ln[:150] + " ...)" if len(ln) > 150 else ln))  # the query vector literal is 384 numbers


VEC_SQL = "SELECT id FROM memories ORDER BY embedding <=> %s::vector LIMIT 20"
FTS_SQL = (
    "SELECT id FROM memories, to_tsquery('english', %s) q WHERE tsv @@ q ORDER BY ts_rank_cd(tsv, q) DESC LIMIT 20"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=10)
    ap.add_argument("--per-user", type=int, default=2000)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    users = bench_users(args.users)
    print(f"loading {args.users} users x {args.per_user} rows ...")
    load(users, args.per_user)

    q = np.random.default_rng(1).standard_normal(DIM).astype(np.float32)
    qv = to_pgvector((q / np.linalg.norm(q)).tolist())
    me = users[0]
    try:
        with connect() as conn:  # app_user, RLS on
            total = conn.execute("SELECT count(*) FROM pg_class WHERE relname = 'memories'").fetchone()
            assert total
            with user_session(conn, me):
                plan, ms = explain(conn, VEC_SQL, (qv,))
                rows = len(conn.execute(VEC_SQL, (qv,)).fetchall())
                show("vector, HNSW default (ef_search=40)", plan, ms, rows)

            with user_session(conn, me):
                conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
                plan, ms = explain(conn, VEC_SQL, (qv,))
                rows = len(conn.execute(VEC_SQL, (qv,)).fetchall())
                show("vector, HNSW + iterative_scan=relaxed_order", plan, ms, rows)

            with user_session(conn, me):
                conn.execute("SET LOCAL enable_indexscan = off")
                plan, ms = explain(conn, VEC_SQL, (qv,))
                rows = len(conn.execute(VEC_SQL, (qv,)).fetchall())
                show("vector, index scans off (exact)", plan, ms, rows)

            with user_session(conn, me):
                plan, ms = explain(conn, FTS_SQL, ("zanzibar",))
                rows = len(conn.execute(FTS_SQL, ("zanzibar",)).fetchall())
                show("full-text 'zanzibar' (rare word)", plan, ms, rows)

            with user_session(conn, me):
                plan, ms = explain(conn, FTS_SQL, ("coffee",))
                rows = len(conn.execute(FTS_SQL, ("coffee",)).fetchall())
                show("full-text 'coffee' (common word)", plan, ms, rows)
    finally:
        if not args.keep:
            with psycopg.connect(ADMIN_DSN, autocommit=True) as admin:
                admin.execute("DELETE FROM memories WHERE user_id = ANY(%s::uuid[])", (users,))
                admin.execute("ANALYZE memories")
            print("\nbench rows removed")


if __name__ == "__main__":
    main()
