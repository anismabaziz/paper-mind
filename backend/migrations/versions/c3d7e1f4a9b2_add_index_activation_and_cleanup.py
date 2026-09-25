"""Record the activation time and the durable generation cleanup work."""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c3d7e1f4a9b2"
down_revision: Union[str, Sequence[str], None] = "b8e4c1a97d30"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the activation time and the cleanup work table."""
    op.add_column(
        "files",
        sa.Column("index_activated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_table(
        "index_generation_cleanups",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("file_id", sa.String(length=32), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "state", sa.String(length=16), nullable=False, server_default="pending"
        ),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["file_id"], ["files.id"], ondelete="CASCADE"),
    )
    op.create_index(
        "ix_index_generation_cleanups_file_id",
        "index_generation_cleanups",
        ["file_id"],
    )
    op.create_index(
        "ix_index_generation_cleanups_filename",
        "index_generation_cleanups",
        ["filename"],
    )
    op.create_index(
        "uq_index_generation_cleanups_target",
        "index_generation_cleanups",
        ["file_id", "generation"],
        unique=True,
    )


def downgrade() -> None:
    """Remove the activation time and the cleanup work table."""
    op.drop_index(
        "uq_index_generation_cleanups_target",
        table_name="index_generation_cleanups",
    )
    op.drop_index(
        "ix_index_generation_cleanups_filename",
        table_name="index_generation_cleanups",
    )
    op.drop_index(
        "ix_index_generation_cleanups_file_id",
        table_name="index_generation_cleanups",
    )
    op.drop_table("index_generation_cleanups")
    op.drop_column("files", "index_activated_at")
