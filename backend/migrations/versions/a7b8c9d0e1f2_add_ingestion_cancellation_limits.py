"""Add cancellation state and ingestion resource tracking."""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a7b8c9d0e1f2"
down_revision: Union[str, Sequence[str], None] = "f2b4c6d8a013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add durable cancellation and resource fields."""
    op.add_column(
        "files",
        sa.Column("index_generation", sa.Integer(), nullable=True),
    )
    op.add_column(
        "ingestion_jobs",
        sa.Column(
            "cancel_requested_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.add_column(
        "ingestion_jobs",
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "ingestion_jobs",
        sa.Column(
            "limits_json",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
    )
    op.add_column(
        "ingestion_jobs",
        sa.Column(
            "usage_json",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
    )
    op.execute("DROP INDEX IF EXISTS uq_ingestion_jobs_one_active")
    op.execute(
        "CREATE UNIQUE INDEX uq_ingestion_jobs_one_active "
        "ON ingestion_jobs (file_id) "
        "WHERE state IN ('queued', 'running', 'cancelling')"
    )


def downgrade() -> None:
    """Remove durable cancellation and resource fields."""
    op.execute("DROP INDEX IF EXISTS uq_ingestion_jobs_one_active")
    op.execute(
        "CREATE UNIQUE INDEX uq_ingestion_jobs_one_active "
        "ON ingestion_jobs (file_id) "
        "WHERE state IN ('queued', 'running')"
    )
    op.drop_column("ingestion_jobs", "usage_json")
    op.drop_column("ingestion_jobs", "limits_json")
    op.drop_column("ingestion_jobs", "cancelled_at")
    op.drop_column("ingestion_jobs", "cancel_requested_at")
    op.drop_column("files", "index_generation")
