"""
create the legacy user_settings table.

Revision ID: f1c3a9b4d2e7
Revises: 8d1f0c2ba743
Create Date: 2026-09-05 10:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "f1c3a9b4d2e7"
down_revision: Union[str, Sequence[str], None] = "8d1f0c2ba743"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

DEMO_USER_ID = "13db88b6ca6a499b8f1dbd3d504e1b83"
DEMO_EMAIL = "demo@papermind.local"
DEMO_PASSWORD_HASH = "$2b$12$XCQV9kxyNy.WmRGKqcaNsOcAErzCcotZcHvOT/JtTpPAJVhY3QU8."


def upgrade() -> None:
    """Upgrade schema."""
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

    bind = op.get_bind()
    exists = bind.execute(
        sa.text("SELECT 1 FROM users WHERE email = :email"),
        {"email": DEMO_EMAIL},
    ).scalar()
    if not exists:
        bind.execute(
            sa.text(
                "INSERT INTO users (id, email, password_hash) "
                "VALUES (:id, :email, :password_hash)"
            ),
            {
                "id": DEMO_USER_ID,
                "email": DEMO_EMAIL,
                "password_hash": DEMO_PASSWORD_HASH,
            },
        )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("user_settings")
