CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
    id          text PRIMARY KEY,
    user_id     uuid        NOT NULL,
    text        text        NOT NULL,
    embedding   vector(384) NOT NULL,          -- BAAI/bge-small-en-v1.5, same as jobs-mcp/rank.py
    importance  real        NOT NULL DEFAULT 0.5,
    created_at  timestamptz NOT NULL DEFAULT now(),
    -- generated column: Postgres recomputes it whenever text changes, nobody writes it by hand
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE INDEX IF NOT EXISTS memories_embedding_hnsw ON memories USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS memories_tsv_gin        ON memories USING gin (tsv);
CREATE INDEX IF NOT EXISTS memories_user_id        ON memories (user_id);

-- ---------------------------------------------------------------------------
-- Row Level Security
--
-- The app never connects as `memory` (the owner / superuser, which skips RLS).
-- It connects as `app_user`: not the owner, NOBYPASSRLS, only DML grants.
-- Every request runs in a transaction that first does
--     SELECT set_config('app.user_id', '<uuid>', true)   -- same as SET LOCAL
-- and the policy below filters every row by it, so a query without any
-- WHERE user_id = ... still only sees that user's rows.
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'app_user') THEN
        -- local dev credential only, matches APP_DSN in memory_service/db.py
        CREATE ROLE app_user LOGIN PASSWORD 'app_user' NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
    END IF;
END $$;

GRANT CONNECT ON DATABASE memory TO app_user;
GRANT USAGE ON SCHEMA public TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON memories TO app_user;

ALTER TABLE memories ENABLE ROW LEVEL SECURITY;
ALTER TABLE memories FORCE ROW LEVEL SECURITY;  -- also applies to the owner if it is ever not a superuser

-- nullif(...): with no app.user_id set, current_setting(..., true) gives '' or NULL;
-- ''::uuid would raise, NULL just matches nothing -> forgetting SET LOCAL returns 0 rows, not an error.
DROP POLICY IF EXISTS memories_owner ON memories;
CREATE POLICY memories_owner ON memories
    USING      (user_id = nullif(current_setting('app.user_id', true), '')::uuid)   -- what you can read / update / delete
    WITH CHECK (user_id = nullif(current_setting('app.user_id', true), '')::uuid);  -- what you can write
