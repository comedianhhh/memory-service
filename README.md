# memory-service

Per-user memory store on Postgres: pgvector cosine search + `tsvector` full-text search, merged with Reciprocal Rank Fusion.

```
query ─┬─ embed (bge-small-en-v1.5, 384d) ─ ORDER BY embedding <=> q  (HNSW) ─ top 20 ─┐
       └─ to_tsquery('english', w1 | w2 …) ─ WHERE tsv @@ q ORDER BY ts_rank_cd (GIN) ─ top 20 ─┴─ RRF k=60 ─ top 5
```

- `tsv` is a generated column (`GENERATED ALWAYS AS (to_tsvector('english', text)) STORED`), so it never drifts from `text`.
- RRF (`memory_service/rrf.py`) fuses ranked id lists, not score arrays, because each SQL retriever only returns its own top-N.
- Full-text query words are OR-ed: `plainto_tsquery` ANDs every word and almost never matches a natural-language question.

## Run

```bash
docker compose up -d
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m pytest -q
.venv/Scripts/python scripts/seed.py
.venv/Scripts/python eval/run.py
```

## Eval (2026-11-03, 40 memories, 20 labelled queries)

| method | recall@5 | MRR |
|---|---|---|
| vector | 0.950 | 0.967 |
| full-text | 0.525 | 0.525 |
| RRF | 0.975 | 0.942 |

Small corpus: 5 of 40 is already 12.5% of everything, so these numbers are optimistic. Target is 50 queries next week, then a reranker.

## Not yet

RLS + non-owner role, FastAPI endpoints, Alembic, JSON timing/token logs, `EXPLAIN ANALYZE` on a bigger table.
