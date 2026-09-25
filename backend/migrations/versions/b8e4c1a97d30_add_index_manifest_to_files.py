"""Record the index manifest and its staleness for stored documents."""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b8e4c1a97d30"
down_revision: Union[str, Sequence[str], None] = "a7b8c9d0e1f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the index manifest and its stale reason to documents."""
    op.add_column(
        "files",
        sa.Column("index_manifest", sa.Text(), nullable=True),
    )
    op.add_column(
        "files",
        sa.Column("index_stale_reason", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    """Remove the index manifest and its stale reason."""
    op.drop_column("files", "index_stale_reason")
    op.drop_column("files", "index_manifest")
