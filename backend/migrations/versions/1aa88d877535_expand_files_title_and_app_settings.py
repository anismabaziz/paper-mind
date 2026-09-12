"""
expand files with title/original_filename and create app_settings.

Revision ID: 1aa88d877535
Revises: f313b2142f4a
Create Date: 2026-09-08
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "1aa88d877535"
down_revision: Union[str, Sequence[str], None] = "f313b2142f4a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

APP_SETTINGS_ID = "app"


def _seed_app_settings(bind) -> None:
    """Copy demo@papermind.local user_settings into app_settings if present."""
    try:
        demo_row = bind.execute(
            sa.text(
                """
                SELECT us.provider, us.model, us.encrypted_api_key
                FROM user_settings us
                JOIN users u ON u.id = us.user_id
                WHERE u.email = :email
                LIMIT 1
                """
            ),
            {"email": "demo@papermind.local"},
        ).mappings().first()

        if not demo_row:
            return

        existing = bind.execute(sa.text("SELECT 1 FROM app_settings LIMIT 1")).scalar()
        if existing:
            return

        bind.execute(
            sa.text(
                """
                INSERT INTO app_settings (id, provider, model, encrypted_api_key)
                VALUES (:id, :provider, :model, :encrypted_api_key)
                """
            ),
            {
                "id": APP_SETTINGS_ID,
                "provider": demo_row["provider"],
                "model": demo_row["model"],
                "encrypted_api_key": demo_row["encrypted_api_key"],
            },
        )
    except Exception as exc:  # best-effort: schema must succeed even if seed fails
        # Don't fail the migration if seed data is missing or tables differ.
        print(f"app_settings seed skipped: {exc}")


def upgrade() -> None:
    """Add title columns to files, create app_settings, seed from demo row."""
    # -- schema (expand-only) --
    op.add_column("files", sa.Column("title", sa.Text(), nullable=True))
    op.add_column(
        "files", sa.Column("original_filename", sa.String(length=255), nullable=True)
    )

    op.create_table(
        "app_settings",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("encrypted_api_key", sa.Text(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # -- data (separate concern, best-effort) --
    bind = op.get_bind()
    _seed_app_settings(bind)


def downgrade() -> None:
    """Remove app_settings and files title columns."""
    op.drop_table("app_settings")
    op.drop_column("files", "original_filename")
    op.drop_column("files", "title")
