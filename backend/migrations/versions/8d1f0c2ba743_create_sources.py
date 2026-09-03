"""create sources

Revision ID: 8d1f0c2ba743
Revises: 4206402c3f83
Create Date: 2026-09-04

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8d1f0c2ba743'
down_revision: Union[str, Sequence[str], None] = '4206402c3f83'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('sources',
    sa.Column('id', sa.String(length=32), nullable=False),
    sa.Column('message_id', sa.String(length=32), nullable=False),
    sa.Column('content', sa.String(length=8192), nullable=False),
    sa.Column('document', sa.String(length=255), nullable=False),
    sa.Column('chunk_index', sa.Integer(), nullable=False),
    sa.Column('score', sa.Float(), nullable=False),
    sa.ForeignKeyConstraint(['message_id'], ['messages.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('sources')
