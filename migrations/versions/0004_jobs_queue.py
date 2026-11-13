"""jobs table: a Postgres work queue for async message ingestion, and the worker role

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-13

Why a table and not SQS: the API must store "message accepted" and "work to do" together. Postgres
plus SQS is two systems with no shared transaction (commit then send can fail half way: the dual-write
problem). A job row commits in the same transaction as everything else, so it can't be lost.
Same semantics as SQS: at-least-once delivery, a visibility timeout (lease), retries, a dead state.
"""

import os

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

WORKER_PASSWORD = os.environ.get("WORKER_DB_PASSWORD", "worker")  # local dev default


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE jobs (
            id           bigserial PRIMARY KEY,
            user_id      uuid        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            kind         text        NOT NULL,
            -- fixed at enqueue time, so every retry writes the same memory ids ({message_id}-{i})
            message_id   text        NOT NULL UNIQUE,
            payload      jsonb       NOT NULL,
            status       text        NOT NULL DEFAULT 'queued'
                         CHECK (status IN ('queued', 'running', 'done', 'dead')),
            attempts     int         NOT NULL DEFAULT 0,
            max_attempts int         NOT NULL DEFAULT 5,
            -- SQS's visibility timeout: a claimed job is invisible until visible_at, then claimable again
            visible_at   timestamptz NOT NULL DEFAULT now(),
            -- fencing token: changes on every claim, so a worker whose lease expired can't complete it
            lease        uuid,
            last_error   text,
            result       jsonb,
            created_at   timestamptz NOT NULL DEFAULT now(),
            finished_at  timestamptz
        )
        """
    )
    # The claim query only looks at claimable rows, oldest first; a partial index keeps that cheap
    # even when millions of done rows pile up.
    op.execute("CREATE INDEX jobs_claimable ON jobs (visible_at, id) WHERE status IN ('queued', 'running')")
    op.execute("CREATE INDEX jobs_user_id ON jobs (user_id)")

    # Worker role: claims jobs for every user, but writes memories under the job owner's RLS session.
    op.execute(
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'worker') THEN
                CREATE ROLE worker LOGIN PASSWORD '{WORKER_PASSWORD}'
                    NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
            END IF;
            EXECUTE format('GRANT CONNECT ON DATABASE %I TO worker', current_database());
        END $$
        """
    )
    op.execute("GRANT USAGE ON SCHEMA public TO worker")
    op.execute("GRANT SELECT, UPDATE ON jobs TO worker")
    op.execute("GRANT SELECT, INSERT ON memories TO worker")  # memories_owner RLS still applies
    op.execute("GRANT SELECT, INSERT ON jobs TO app_user")
    op.execute("GRANT USAGE ON SEQUENCE jobs_id_seq TO app_user")

    op.execute("ALTER TABLE jobs ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE jobs FORCE ROW LEVEL SECURITY")
    # API users see and create only their own jobs ...
    op.execute(
        """
        CREATE POLICY jobs_owner ON jobs TO app_user
            USING      (user_id = nullif(current_setting('app.user_id', true), '')::uuid)
            WITH CHECK (user_id = nullif(current_setting('app.user_id', true), '')::uuid)
        """
    )
    # ... the worker sees the whole queue.
    op.execute("CREATE POLICY jobs_worker ON jobs TO worker USING (true) WITH CHECK (true)")


def downgrade() -> None:
    op.execute("DROP TABLE jobs")
    op.execute("REVOKE ALL ON memories FROM worker")
    op.execute("REVOKE ALL ON SCHEMA public FROM worker")
    op.execute("DO $$ BEGIN EXECUTE format('REVOKE CONNECT ON DATABASE %I FROM worker', current_database()); END $$")
