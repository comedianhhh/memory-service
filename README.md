# memory-service

Per-user memory store on Postgres: pgvector cosine search + `tsvector` full-text search, merged with Reciprocal Rank Fusion. Tenant isolation is enforced by Postgres Row Level Security, not by `WHERE` clauses.

```
POST /users/{id}/messages ─► chunk (≤80 words, 1-sentence overlap) ─► embed batch ─► INSERT n rows (RLS WITH CHECK)
POST /users/{id}/memories ─► embed ─► INSERT 1 row

GET  /users/{id}/memories/search?q=&mode=hybrid|vector|fts&k=5
  connection from pool (app_user)
  BEGIN; set_config('app.user_id', id, true)           -- = SET LOCAL
  query ─┬─ embed (bge-small-en-v1.5, 384d) ─ HNSW, iterative scan ─ top 20 ─┐
         └─ to_tsquery('english', w1 | w2 …) ─ tsv @@ q, ts_rank_cd ─ top 20 ─┴─ RRF k=60 ─ top k
  COMMIT                                                -- setting dies with the transaction
  connection back to pool (RESET ALL)
```

## Design notes

**Row Level Security**
- The app connects as `app_user`: not the table owner, `NOBYPASSRLS`, only DML grants. `memory` is a superuser in the Docker image, and superusers always skip RLS, so it's only used for migrations, tests and bulk loads.
- The policy is `user_id = nullif(current_setting('app.user_id', true), '')::uuid` for both `USING` (read / update / delete) and `WITH CHECK` (write). A query that forgets the setting sees **0 rows**, not everyone's.
- `tests/test_rls.py` has no user filter anywhere. With RLS disabled, 4 of its 6 tests fail.

**Connection pool** (`psycopg_pool`, 1–10 connections)
- Each request borrows a connection; on return the pool runs `RESET ALL`.
- `user_session` already uses `SET LOCAL`, so nothing should outlive the transaction. The reset is a second line of defence against a future session-level `SET`.
- `tests/test_pool.py` forces one physical connection to serve user A and then user B. With the reset removed, that test fails.

**Chunking** (`memory_service/chunk.py`)
- One vector per long message averages every topic in it. So messages are packed into whole sentences up to 80 words, roughly 100 tokens, well under bge-small's 512-token limit.
- The last sentence of each chunk is repeated at the start of the next, so a fact on a boundary stays intact.
- Each chunk keeps `source_id` and `chunk_index`, so it can be traced back to its message.

**HNSW + RLS gotcha (found by the benchmark, fixed)**
- HNSW returns the `ef_search` (40) nearest rows from the **whole table**, and only then does RLS drop other users' rows.
- In a 20k-row table with 10 users, a top-20 query came back with **6 rows**. The regression test reproduced it with **0 rows**.
- `vector_search` now sets `hnsw.iterative_scan = relaxed_order` (pgvector ≥ 0.8), so the index keeps searching until LIMIT rows pass the filter. A materialized CTE then re-sorts the results exactly.

**Migrations.** Alembic, plain SQL, run as the owner: `0001` creates the table, indexes, role and RLS; `0002` adds `source_id` and `chunk_index`. Both downgrade cleanly.

**Logs.** One JSON object per line on stderr: `chunk`, `embed` (with real tokenizer `tokens`), `retrieve`, `fuse`, `search`, `db.insert`, `http`, each with `duration_ms`.

## EXPLAIN ANALYZE (`scripts/bench_index.py`: 10 users × 2000 rows, run as app_user under RLS)

Two runs, local Docker on Windows:

| query | plan | rows back | time |
|---|---|---|---|
| vector top 20, HNSW default | Index Scan using `memories_embedding_hnsw`, the rest removed by the RLS filter | **6, 4** | 3.0–3.2 ms |
| vector top 20, HNSW + iterative scan | Index Scan using `memories_embedding_hnsw`, ~170 removed | 20 | 4.2–11.9 ms |
| vector top 20, index scans off | Bitmap scan on `memories_user_id` + top-N sort (exact) | 20 | 2.4–3.6 ms |
| full-text `zanzibar` (rare) | Bitmap scan on `memories_user_id`, then `tsv @@ q` filter | 4 | 0.3–0.4 ms |
| full-text `coffee` (common) | same | 20 | 0.5–0.7 ms |

What the plans show:
- **Full-text searches don't use the GIN index at this size.** The planner filters by `user_id` first: 2000 rows is smaller than any GIN result, so that's the right call. GIN starts to pay off once a single user has many rows.
- **Exact search over one user's 2000 rows is as fast as HNSW, often faster (2.4–3.6 vs 4.2–11.9 ms), and has perfect recall.** For per-user memory, where each user has thousands of rows rather than millions, the exact scan via `user_id` is a valid design. HNSW earns its keep only when one user's rows are huge.

## Run

```bash
docker compose up -d --wait
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
.venv/Scripts/alembic upgrade head
.venv/Scripts/python -m pytest -q          # 27 tests; DB tests skip if Postgres is down
.venv/Scripts/python scripts/seed.py
.venv/Scripts/python eval/run.py
.venv/Scripts/python scripts/bench_index.py
.venv/Scripts/uvicorn memory_service.api:app --port 8000
```

Tests use `FakeEmbedder` (hashed bag of words), so they need no model download.

## Eval (40 memories, 20 labelled queries)

| method | recall@5 | MRR |
|---|---|---|
| vector | 0.950 | 0.967 |
| full-text | 0.525 | 0.525 |
| RRF | 0.975 | 0.942 |

Small corpus: 5 of 40 is already 12.5% of everything, so these numbers are optimistic. Target is 50 queries next week, then a reranker.

## Known gap

`user_id` comes from the URL path. With real auth it must come from the verified token, otherwise anyone can pass someone else's id.
