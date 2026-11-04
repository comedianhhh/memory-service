# memory-service

Per-user memory store on Postgres: pgvector cosine search + `tsvector` full-text search, merged with Reciprocal Rank Fusion, optionally reranked by a cross-encoder. Users log in with a password and get short-lived access tokens plus rotating refresh tokens. Tenant isolation is enforced by Postgres Row Level Security, not by `WHERE` clauses.

```
POST /auth/register | /auth/login | /auth/refresh | /auth/logout      GET /me

POST /me/messages ─► chunk (≤80 words, 1-sentence overlap) ─► embed batch ─► INSERT n rows (RLS WITH CHECK)
POST /me/memories ─► embed ─► INSERT 1 row

GET  /me/memories/search?q=&mode=hybrid|vector|fts&k=5
  Bearer token ─► verify JWT (HS256 pinned, exp) ─► session row active? ─► user_id
  connection from pool (app_user, liveness-checked)
  BEGIN; set_config('app.user_id', user_id, true)      -- = SET LOCAL
  query ─┬─ embed (bge-small-en-v1.5, 384d) ─ HNSW, iterative scan ─ top 20 ─┐
         └─ to_tsquery('english', w1 | w2 …) ─ tsv @@ q, ts_rank_cd ─ top 20 ─┴─ RRF k=60 ─ top k
  COMMIT                                                -- setting dies with the transaction
  connection back to pool (RESET ALL)
```

## Auth and session lifetime

| what | lifetime | form | checked |
|---|---|---|---|
| access token | 15 min | JWT (`sub`, `sid`, `exp`), HS256 | every request: signature, exp, and the session row |
| refresh token | 7 days idle | 32 random bytes; stored only as sha256 | single use: each refresh burns it and returns a new one |
| session | 30 days absolute | `sessions` row | refresh can never extend past it; logout sets `revoked_at` |

- **User identity comes only from the token.** No URL carries a user id, so there is nothing to tamper with. Editing `sub` in a correctly signed token is rejected too, because it must match the session's user.
- **Every request checks the session row.** It costs one indexed lookup, and in return logout works immediately rather than up to 15 minutes later.
- **Refresh token reuse.** A refresh token presented a second time revokes the whole session: either the client replayed it, or it was stolen and both copies are in use, and the server can't tell which copy is the attacker's.
- **Bug caught by `test_reusing_a_refresh_token_kills_the_whole_session`.** The revoke was originally inside the transaction that then raised, so the rollback undid it. Errors are now raised after the commit.
- **Passwords** are argon2id. Login for an unknown email still runs a dummy verify, so it takes as long as a wrong password and returns the same `invalid_credentials`.
- **`alg: none` and tokens signed with other keys are rejected:** the algorithm is pinned server-side.
- **401 bodies carry an `error` code**, so a client knows whether refreshing can help (`token_expired`) or it must log in again (`session_expired`, `session_revoked`, `refresh_token_reused`, …).
- **Testing time-based behaviour.** Every auth function takes `now`, and tests inject a clock instead of sleeping 30 days.

## Reconnect

**Server → Postgres** (`memory_service/db.py`)
- The pool pings each connection on checkout (`check=ConnectionPool.check_connection`) and replaces dead ones.
- It also recycles connections after an hour and keeps retrying in the background for 60 s if the database is down.
- `tests/test_reconnect.py` terminates every `app_user` backend from the server side, and the next requests still return 200. Without `check` that test fails with "connection abort".

**Client → API** (`memory_service/client.py`, `MemoryClient`)
- On `401 token_expired` it refreshes once and retries the request once.
- When refresh is no longer possible it raises `SessionExpired`.
- Refresh runs behind a lock that skips the call if another thread already refreshed. Refresh tokens are single use, so a double refresh would look like theft and get the session revoked.
- Connection errors are retried with backoff, but only for idempotent methods. A retried POST could store a memory twice.

## Row Level Security

- The app connects as `app_user`: not the table owner, `NOBYPASSRLS`, only DML grants. `memory` is a superuser in the Docker image, and superusers always skip RLS, so it's only used for migrations, tests and bulk loads.
- The policy on `memories` is `user_id = nullif(current_setting('app.user_id', true), '')::uuid` for both `USING` and `WITH CHECK`. A query that forgets the setting sees **0 rows**.
- `tests/test_rls.py` has no user filter anywhere. With RLS disabled, 4 of its 6 tests fail.
- The auth tables (`users`, `sessions`, `refresh_tokens`) have no RLS: login must find a user by email before anyone is authenticated. Only `auth.py` touches them.
- On return to the pool, connections get `RESET ALL`. `tests/test_pool.py` reuses one physical connection for two users; without the reset that test fails.

## Chunking

- One vector per long message averages every topic in it. So messages are packed into whole sentences up to 80 words, roughly 100 tokens, under bge-small's 512-token limit.
- The last sentence of each chunk is repeated at the start of the next, so a fact on a boundary stays intact.
- Each chunk keeps `source_id` and `chunk_index`.

## HNSW + RLS gotcha (found by the benchmark, fixed)

- HNSW returns the `ef_search` (40) nearest rows from the **whole table**, and only then does RLS drop other users' rows.
- In a 20k-row table with 10 users, a top-20 query came back with **4–6 rows**. The regression test reproduced it with **0**.
- `vector_search` now sets `hnsw.iterative_scan = relaxed_order` (pgvector ≥ 0.8) and re-sorts in a materialized CTE.

## EXPLAIN ANALYZE (`scripts/bench_index.py`: 10 users × 2000 rows, as app_user under RLS)

Two runs, local Docker on Windows:

| query | plan | rows back | time |
|---|---|---|---|
| vector top 20, HNSW default | Index Scan using `memories_embedding_hnsw`, the rest removed by the RLS filter | **6, 4** | 3.0–3.2 ms |
| vector top 20, HNSW + iterative scan | Index Scan using `memories_embedding_hnsw`, ~170 removed | 20 | 4.2–11.9 ms |
| vector top 20, index scans off | Bitmap scan on `memories_user_id` + top-N sort (exact) | 20 | 2.4–3.6 ms |
| full-text `zanzibar` (rare) | Bitmap scan on `memories_user_id`, then `tsv @@ q` filter | 4 | 0.3–0.4 ms |
| full-text `coffee` (common) | same | 20 | 0.5–0.7 ms |

What the plans show:
- **Full-text searches don't use GIN at this size.** Filtering by `user_id` first leaves only 2000 rows, so that's the right call.
- **Exact search over one user's 2000 rows is as fast as HNSW, often faster, and has perfect recall.** For per-user memory, where each user has thousands of rows rather than millions, exact scan via `user_id` is a valid design.

## Eval (`eval/run.py`: 40 memories, 50 labelled queries)

| method | recall@5 | MRR | p50 latency |
|---|---|---|---|
| vector | 0.980 | 0.938 | 5.5 ms |
| full-text | 0.660 | 0.637 | 0.7 ms |
| RRF | 0.990 | 0.923 | 6.3 ms |
| RRF + rerank (`ms-marco-MiniLM-L-6-v2`, top 20) | 0.990 | **0.957** | 34.1 ms |

What the numbers say:
- **Recall@5 is saturated at 0.99.** With 40 memories, 5 results already cover 12.5% of everything, so recall can't separate the methods any more. The corpus needs to grow before recall is informative.
- **RRF lifts recall over vector alone, but lowers MRR** (0.938 → 0.923). Weak full-text rankings push the right answer down a place.
- **The reranker wins that back and more** (0.957): the right answer is first more often. It costs ~28 ms per query on CPU.
- **The one query nothing fixes is q10**, "is it safe to keep training for the race". It needs "physio told her to cut mileage", which is reasoning, not retrieval.

## Chunking eval on conversations (`eval/convo_run.py`)

**Setup**
- Five synthetic conversations, 21–24 turns each (food, work, family, hobbies, money), in `data/conversations/`.
- Each one is stored four ways, each under its own user so RLS keeps them apart.
- Labels are **facts**, not ids, because ids change with chunking. A fact counts only if all of its phrases sit in one retrieved chunk. For example, "what's your partner's name?" + "Theo." must land together: a bare "Noor: Theo." can't be found by "partner's name", and wouldn't tell the model whose name it is.
- About 10 of the 26 queries are like that on purpose; real chat answers are often just "January." or "Pickle."

| chunking | chunks | avg words | RRF recall@5 | RRF MRR | +rerank recall@5 | +rerank MRR | words sent to LLM (top 5) |
|---|---|---|---|---|---|---|---|
| one turn per memory | 114 | 7 | 0.577 | 0.389 | 0.519 | 0.480 | 36 |
| ≤40 words | 28 | 33 | 1.000 | 0.933 | 1.000 | 0.962 | 167 |
| ≤80 words (API default) | 13 | 67 | 1.000 | 0.910 | 1.000 | 0.955 | 344 |
| ≤160 words | 8 | 107 | 1.000 | 0.936 | 1.000 | 0.981 | 623 |

What it shows:
- **Storing each turn alone is the real failure.** Recall@5 is 0.58: answers lose their question, and even single-phrase facts like "low ferritin" can't be reached from "why does she need more iron" without the surrounding turn.
- **Rerank made per-turn recall worse.** A cross-encoder can't fix chunks that don't contain enough to judge.
- **Bigger chunks look free here but aren't.** With 8 chunks in total, top 5 is 62% of the data, so 160 words "wins" by returning almost everything. The real cost is the last column: 623 words go into the prompt instead of 167.
- **On this data ≤40 words is the sweet spot:** full recall at a quarter of the 160-word context.
- **Caveat:** 5 conversations is tiny. The chunk size should be re-checked when there are hundreds per user.

## Run

```bash
docker compose up -d
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m pytest -q          # 49 tests; waits for Postgres and runs migrations itself
.venv/Scripts/alembic upgrade head         # only needed before running the scripts below without pytest first
.venv/Scripts/python scripts/seed.py
.venv/Scripts/python eval/run.py
.venv/Scripts/python eval/convo_run.py
.venv/Scripts/python scripts/bench_index.py
JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(48))") .venv/Scripts/uvicorn memory_service.api:app --port 8000
```

```python
from memory_service.client import MemoryClient
mc = MemoryClient("http://localhost:8000")
mc.login("you@example.com", "a long password")
mc.remember("Biscuit needs thyroid pills twice a day")
mc.search("pet medication")       # refreshes the access token by itself when it expires
```

Tests use `FakeEmbedder` (hashed bag of words) and an injected clock, so they need no model download and no waiting.

## Not yet

- Login rate limiting / lockout.
- Email verification and password reset.
- Reranking as an API option. It's in the eval only so far.
