"""
Durable document-deletion state on files.

Revision ID: e7a1c3d5f908
Revises: c4d2e9a1b5f6
Create Date: 2026-09-24
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e7a1c3d5f908"
down_revision: Union[str, Sequence[str], None] = "c4d2e9a1b5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Track durable deletion state so cleanup can be retried."""
    op.add_column(
        "files",
        sa.Column(
            "deletion_state",
            sa.String(16),
            nullable=False,
            server_default="active",
        ),
    )
    op.add_column("files", sa.Column("deletion_error", sa.Text(), nullable=True))
    op.add_column(
        "files",
        sa.Column(
            "deletion_attempts",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    """Remove durable deletion state."""
    op.drop_column("files", "deletion_attempts")
    op.drop_column("files", "deletion_error")
    op.drop_column("files", "deletion_state")
