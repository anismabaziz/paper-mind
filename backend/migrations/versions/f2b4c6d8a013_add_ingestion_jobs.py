"""
Durable ingestion jobs for stored documents.

Revision ID: f2b4c6d8a013
Revises: e7a1c3d5f908
Create Date: 2026-09-24
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f2b4c6d8a013"
down_revision: Union[str, Sequence[str], None] = "e7a1c3d5f908"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Track durable ingestion jobs so uploads survive worker restarts."""
    op.create_table(
        "ingestion_jobs",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "file_id",
            sa.String(32),
            sa.ForeignKey("files.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("filename", sa.String(255), nullable=False, index=True),
        sa.Column("generation", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("state", sa.String(16), nullable=False, server_default="queued"),
        sa.Column("stage", sa.String(32), nullable=False, server_default="queued"),
        sa.Column("progress", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("error_category", sa.String(64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("worker_id", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
    )
    # One active (queued or running) job per document.
    op.execute(
        "CREATE UNIQUE INDEX uq_ingestion_jobs_one_active "
        "ON ingestion_jobs (file_id) "
        "WHERE state IN ('queued', 'running')"
    )


def downgrade() -> None:
    """Remove durable ingestion jobs."""
    op.execute("DROP INDEX IF EXISTS uq_ingestion_jobs_one_active")
    op.drop_table("ingestion_jobs")
