"""
Keep the reason an answer was abstained on the turn itself.

Revision ID: b3a7c9e1f4d2
Revises: e5b7d2c4a918
Create Date: 2026-09-26
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b3a7c9e1f4d2"
down_revision: Union[str, Sequence[str], None] = "e5b7d2c4a918"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Keep the reason an answer was abstained on the turn itself."""
    with op.batch_alter_table("turns") as batch:
        batch.add_column(
            sa.Column("abstention_reason", sa.String(length=64), nullable=True)
        )


def downgrade() -> None:
    """Drop the abstention reason, leaving the answer and its status."""
    with op.batch_alter_table("turns") as batch:
        batch.drop_column("abstention_reason")
