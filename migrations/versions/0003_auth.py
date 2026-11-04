"""users, login sessions, rotating refresh tokens

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-04

No RLS on these three tables: login has to find a user by email before anyone is authenticated.
They are only touched by memory_service/auth.py. Isolation of user data stays on `memories`.
"""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE users (
            id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            email          text NOT NULL UNIQUE CHECK (email = lower(email)),
            password_hash  text NOT NULL,                 -- argon2id, salt and params inside the string
            created_at     timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    # One row per login. Absolute lifetime: after expires_at the user must log in again, however
    # active they are. Logout / reuse detection sets revoked_at.
    op.execute(
        """
        CREATE TABLE sessions (
            id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id        uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at     timestamptz NOT NULL,
            expires_at     timestamptz NOT NULL,
            revoked_at     timestamptz,
            revoke_reason  text
        )
        """
    )
    op.execute("CREATE INDEX sessions_user_id ON sessions (user_id)")
    # Refresh tokens are stored only as sha256: a DB leak doesn't hand out live tokens.
    # Each one is single use (used_at); presenting a used one again means it was stolen -> revoke session.
    op.execute(
        """
        CREATE TABLE refresh_tokens (
            token_hash  bytea PRIMARY KEY,
            session_id  uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            issued_at   timestamptz NOT NULL,
            expires_at  timestamptz NOT NULL,
            used_at     timestamptz
        )
        """
    )
    op.execute("CREATE INDEX refresh_tokens_session_id ON refresh_tokens (session_id)")
    op.execute("GRANT SELECT, INSERT ON users TO app_user")
    op.execute("GRANT SELECT, INSERT, UPDATE ON sessions, refresh_tokens TO app_user")


def downgrade() -> None:
    op.execute("DROP TABLE refresh_tokens")
    op.execute("DROP TABLE sessions")
    op.execute("DROP TABLE users")
