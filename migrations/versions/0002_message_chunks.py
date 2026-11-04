"""source message id + chunk index, so a long message split into chunks can be traced back

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-04
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # nullable: memories written directly (not from a message) have neither
    op.execute("ALTER TABLE memories ADD COLUMN source_id text, ADD COLUMN chunk_index integer")
    op.execute("CREATE INDEX memories_source_id ON memories (source_id) WHERE source_id IS NOT NULL")


def downgrade() -> None:
    op.execute("DROP INDEX memories_source_id")
    op.execute("ALTER TABLE memories DROP COLUMN chunk_index, DROP COLUMN source_id")
