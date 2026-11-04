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
