"""memories table, vector + full-text indexes, app_user role, row level security

Revision ID: 0001
Revises:
Create Date: 2026-11-04
"""

import os

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

# local dev default; set APP_DB_PASSWORD anywhere real
APP_PASSWORD = os.environ.get("APP_DB_PASSWORD", "app_user")


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        """
        CREATE TABLE memories (
            id          text PRIMARY KEY,
            user_id     uuid        NOT NULL,
            text        text        NOT NULL,
            embedding   vector(384) NOT NULL,          -- BAAI/bge-small-en-v1.5, same as jobs-mcp/rank.py
            importance  real        NOT NULL DEFAULT 0.5,
            created_at  timestamptz NOT NULL DEFAULT now(),
            -- generated column: recomputed whenever text changes, nobody writes it by hand
            tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
        )
        """
    )
    op.execute("CREATE INDEX memories_embedding_hnsw ON memories USING hnsw (embedding vector_cosine_ops)")
    op.execute("CREATE INDEX memories_tsv_gin ON memories USING gin (tsv)")
    op.execute("CREATE INDEX memories_user_id ON memories (user_id)")

    # The app connects as app_user: not the owner, NOBYPASSRLS, DML only.
    # Roles are cluster-wide, so create it only if an earlier database already did not.
    op.execute(
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'app_user') THEN
                CREATE ROLE app_user LOGIN PASSWORD '{APP_PASSWORD}'
                    NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
            END IF;
            EXECUTE format('GRANT CONNECT ON DATABASE %I TO app_user', current_database());
        END $$
        """
    )
    op.execute("GRANT USAGE ON SCHEMA public TO app_user")
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON memories TO app_user")

    # Every request does set_config('app.user_id', <uuid>, true) inside its transaction (= SET LOCAL).
    # nullif: with no setting, current_setting(..., true) is '' or NULL; ''::uuid would raise,
    # NULL matches nothing -> forgetting the session setting returns 0 rows, not an error or everything.
    op.execute("ALTER TABLE memories ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE memories FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY memories_owner ON memories
            USING      (user_id = nullif(current_setting('app.user_id', true), '')::uuid)
            WITH CHECK (user_id = nullif(current_setting('app.user_id', true), '')::uuid)
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE memories")
    op.execute("REVOKE ALL ON SCHEMA public FROM app_user")
    op.execute("DO $$ BEGIN EXECUTE format('REVOKE CONNECT ON DATABASE %I FROM app_user', current_database()); END $$")
    # the role itself is cluster-wide and may serve other databases; drop it by hand if wanted
