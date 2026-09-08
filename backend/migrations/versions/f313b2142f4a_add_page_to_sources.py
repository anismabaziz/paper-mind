"""
add page to sources.

Revision ID: f313b2142f4a
Revises: f1c3a9b4d2e7
Create Date: 2026-09-08
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f313b2142f4a"
down_revision: Union[str, Sequence[str], None] = "f1c3a9b4d2e7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add page column to sources for citation page-jump."""
    op.add_column("sources", sa.Column("page", sa.Integer(), nullable=True))


def downgrade() -> None:
    """Remove page column."""
    op.drop_column("sources", "page")
