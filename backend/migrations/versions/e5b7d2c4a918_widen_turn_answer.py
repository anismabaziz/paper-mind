"""
widen the stored answer to unbounded text.

Revision ID: e5b7d2c4a918
Revises: d4f1a8b3c6e2
Create Date: 2026-09-26
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e5b7d2c4a918"
down_revision: Union[str, Sequence[str], None] = "d4f1a8b3c6e2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Let a stored answer run to the model's output budget, not a column width."""
    with op.batch_alter_table("turns") as batch:
        batch.alter_column(
            "answer", existing_type=sa.String(length=8192), type_=sa.Text()
        )


def downgrade() -> None:
    """Restore the fixed-width answer column."""
    with op.batch_alter_table("turns") as batch:
        batch.alter_column(
            "answer",
            existing_type=sa.Text(),
            type_=sa.String(length=8192),
        )
