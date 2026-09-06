"""
create user_settings and seed demo user.

Revision ID: f1c3a9b4d2e7
Revises: 8d1f0c2ba743
Create Date: 2026-09-05 10:00:00.000000
"""

from typing import Sequence, Union

import uuid

from alembic import op
import sqlalchemy as sa

from services.accounts.auth_service import hash_password

# revision identifiers, used by Alembic.
revision: str = "f1c3a9b4d2e7"
down_revision: Union[str, Sequence[str], None] = "8d1f0c2ba743"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

DEMO_EMAIL = "demo@papermind.local"
DEMO_PASSWORD = "demo-password"


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
        sa.text("SELECT 1 FROM users WHERE email = :email"), {"email": DEMO_EMAIL}
    ).scalar()
    if not exists:
        bind.execute(
            sa.text(
                "INSERT INTO users (id, email, password_hash) "
                "VALUES (:id, :email, :password_hash)"
            ),
            {
                "id": uuid.uuid4().hex,
                "email": DEMO_EMAIL,
                "password_hash": hash_password(DEMO_PASSWORD),
            },
        )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("user_settings")
