"""
track last opened time on files.

Revision ID: c4d2e9a1b5f6
Revises: b7c9e2f4a1d6
Create Date: 2026-09-24
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c4d2e9a1b5f6"
down_revision: Union[str, Sequence[str], None] = "b7c9e2f4a1d6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add last_opened_at to files for recent-readings order."""
    op.add_column(
        "files", sa.Column("last_opened_at", sa.DateTime(timezone=True), nullable=True)
    )


def downgrade() -> None:
    """Remove last_opened_at."""
    op.drop_column("files", "last_opened_at")
