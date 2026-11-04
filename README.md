# memory-service

Per-user memory store on Postgres: pgvector cosine search + `tsvector` full-text search, merged with Reciprocal Rank Fusion. Tenant isolation is enforced by Postgres Row Level Security, not by `WHERE` clauses.

```
POST /users/{id}/memories ──► embed ──► INSERT (RLS WITH CHECK)

GET  /users/{id}/memories/search?q=&mode=hybrid|vector|fts&k=5
  BEGIN; set_config('app.user_id', id, true)          -- = SET LOCAL
  query ─┬─ embed (bge-small-en-v1.5, 384d) ─ ORDER BY embedding <=> q   (HNSW) ─ top 20 ─┐
         └─ to_tsquery('english', w1 | w2 …) ─ WHERE tsv @@ q, ts_rank_cd (GIN) ─ top 20 ─┴─ RRF k=60 ─ top k
  COMMIT                                               -- setting dies with the transaction
```

## Design notes

- **RLS.** The app connects as `app_user`: not the table owner, `NOBYPASSRLS`, only DML grants. The policy is `user_id = nullif(current_setting('app.user_id', true), '')::uuid` for both `USING` (read / update / delete) and `WITH CHECK` (write). A query that forgets the session setting sees **0 rows**, not all of them. `FORCE ROW LEVEL SECURITY` is on too. Note that `memory` is a superuser in the Docker image, and superusers always skip RLS, so it is admin-only.
- **Proof it's the database doing it.** `tests/test_rls.py` uses no user filter anywhere. With RLS disabled, 4 of its 6 tests fail.
- **`tsv` is a generated column** (`GENERATED ALWAYS AS (to_tsvector('english', text)) STORED`), so it never drifts from `text`.
- **RRF** (`memory_service/rrf.py`) fuses ranked id lists, not score arrays, because each SQL retriever only returns its own top-N.
- **Full-text query words are OR-ed.** `plainto_tsquery` ANDs every word and almost never matches a natural-language question.
- **Logs** are one JSON object per line on stderr: `embed` (with real tokenizer `tokens`), `retrieve`, `fuse`, `search`, `db.insert`, `http`, each with `duration_ms`.

```json
{"event": "embed", "kind": "query", "model": "BAAI/bge-small-en-v1.5", "n": 1, "tokens": 4, "duration_ms": 491.08}
{"event": "retrieve", "method": "vector", "limit": 20, "hits": 20, "duration_ms": 3.52}
{"event": "retrieve", "method": "fts", "limit": 20, "hits": 0, "duration_ms": 1.82}
```

The first embedding call includes loading the model; later calls are much faster.

## Run

```bash
docker compose up -d --wait
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m pytest -q          # 16 tests; DB tests skip if Postgres is down
.venv/Scripts/python scripts/seed.py
.venv/Scripts/python eval/run.py
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

## Not yet

- `user_id` comes from the URL path. With real auth it must come from the verified token, otherwise anyone can pass someone else's id.
- Chunking long messages, Alembic migrations, connection pooling, `EXPLAIN ANALYZE` on a table big enough for the planner to choose HNSW.
