"""
Keep the claims and the source IDs they cite on the turn.

Revision ID: c7d4e91a5b02
Revises: b3a7c9e1f4d2
Create Date: 2026-09-26
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c7d4e91a5b02"
down_revision: Union[str, Sequence[str], None] = "b3a7c9e1f4d2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Keep the claims and the source IDs they cite on the turn."""
    with op.batch_alter_table("turns") as batch:
        batch.add_column(sa.Column("claims", sa.Text(), nullable=True))
    with op.batch_alter_table("sources") as batch:
        batch.add_column(sa.Column("source_id", sa.String(length=16), nullable=True))
        batch.add_column(sa.Column("rank", sa.Integer(), nullable=True))


def downgrade() -> None:
    """Drop the claims, leaving the answer and the passages it was read from."""
    with op.batch_alter_table("sources") as batch:
        batch.drop_column("rank")
        batch.drop_column("source_id")
    with op.batch_alter_table("turns") as batch:
        batch.drop_column("claims")
