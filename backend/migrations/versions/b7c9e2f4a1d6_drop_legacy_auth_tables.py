"""
drop legacy auth tables.

Revision ID: b7c9e2f4a1d6
Revises: 1aa88d877535
Create Date: 2026-09-12
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "b7c9e2f4a1d6"
down_revision: Union[str, Sequence[str], None] = "1aa88d877535"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Drop legacy multi-user auth tables after app_settings is confirmed."""
    # app_settings is the single global row introduced in 1aa88d877535.
    # Confirm the table exists before dropping legacy tables (expand/contract).
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()
    if "app_settings" not in tables:
        raise RuntimeError(
            "app_settings table missing — run 1aa88d877535 before contracting legacy auth tables"
        )
    # Drop child first due to FK; conditional for idempotence.
    if "user_settings" in tables:
        op.drop_table("user_settings")
    if "users" in tables:
        op.drop_table("users")


def downgrade() -> None:
    """Best-effort recreation of legacy tables (no key reconstruction)."""
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.String(length=128), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )
    op.create_table(
        "user_settings",
        sa.Column("user_id", sa.String(length=32), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("encrypted_api_key", sa.Text(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id"),
    )
