"""
Store one ordered conversation turn per exchange.

Revision ID: d4f1a8b3c6e2
Revises: c3d7e1f4a9b2
Create Date: 2026-09-25
"""

import uuid
from typing import Any, Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d4f1a8b3c6e2"
down_revision: Union[str, Sequence[str], None] = "c3d7e1f4a9b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

ANSWERED = "answered"
UNANSWERED = "unanswered"
NO_ANSWER_RECORDED = "no answer was recorded"
NO_QUESTION_RECORDED = "no question was recorded"


def _merge_duplicate_conversations(connection) -> None:
    """Fold every surplus conversation for a document into the first one."""
    rows = (
        connection.execute(
            sa.text(
                "SELECT id, file_id, created_at FROM conversations "
                "ORDER BY file_id, created_at, id"
            )
        )
        .mappings()
        .all()
    )
    keeper_by_file: dict[str, Any] = {}
    for row in rows:
        keeper = keeper_by_file.setdefault(row["file_id"], row)
        if keeper["id"] == row["id"]:
            continue
        connection.execute(
            sa.text(
                "UPDATE messages SET conversation_id = :keeper "
                "WHERE conversation_id = :surplus"
            ),
            {"keeper": keeper["id"], "surplus": row["id"]},
        )
        if row["created_at"] is not None and (
            keeper["created_at"] is None or row["created_at"] < keeper["created_at"]
        ):
            connection.execute(
                sa.text(
                    "UPDATE conversations SET created_at = :created_at WHERE id = :id"
                ),
                {"created_at": row["created_at"], "id": keeper["id"]},
            )
        connection.execute(
            sa.text("DELETE FROM conversations WHERE id = :id"), {"id": row["id"]}
        )


def _exchanges(rows: Sequence[Any]) -> list[dict[str, Any]]:
    """Split each conversation's messages into exchanges, oldest first."""
    exchanges: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    conversation_id: str | None = None
    for row in rows:
        if row["conversation_id"] != conversation_id:
            conversation_id = row["conversation_id"]
            current = None
        # A user question opens an exchange. A reply that no open question can
        # claim opens an exchange of its own rather than being dropped.
        if current is None or row["sender"] == "user" or current["answer"] is not None:
            current = {
                "conversation_id": conversation_id,
                "question": None,
                "answer": None,
            }
            exchanges.append(current)
        if current["question"] is None:
            current["answer" if row["sender"] != "user" else "question"] = row
        else:
            current["answer"] = row
    return exchanges


def _backfill_turns(connection) -> list[tuple[str, str]]:
    """Write one turn per exchange and report the turn each message became."""
    rows = (
        connection.execute(
            sa.text(
                "SELECT id, conversation_id, sender, text, created_at FROM messages "
                "ORDER BY conversation_id, created_at, id"
            )
        )
        .mappings()
        .all()
    )
    turn_of_message: list[tuple[str, str]] = []
    sequence = 0
    conversation_id: str | None = None
    for exchange in _exchanges(rows):
        if exchange["conversation_id"] != conversation_id:
            conversation_id = exchange["conversation_id"]
            sequence = 0
        sequence += 1
        turn_id = uuid.uuid4().hex
        question = exchange["question"]
        answer = exchange["answer"]
        if question is None:
            status, reason, started_at, completed_at = (
                ANSWERED,
                NO_QUESTION_RECORDED,
                answer["created_at"],
                answer["created_at"],
            )
        elif answer is None:
            status, reason, started_at, completed_at = (
                UNANSWERED,
                NO_ANSWER_RECORDED,
                question["created_at"],
                question["created_at"],
            )
        else:
            status, reason, started_at, completed_at = (
                ANSWERED,
                None,
                question["created_at"],
                answer["created_at"],
            )
        connection.execute(
            sa.text(
                "INSERT INTO turns "
                "(id, conversation_id, sequence, question, answer, status, "
                "failure_reason, created_at, completed_at) "
                "VALUES (:id, :conversation_id, :sequence, :question, :answer, "
                ":status, :reason, :created_at, :completed_at)"
            ),
            {
                "id": turn_id,
                "conversation_id": exchange["conversation_id"],
                "sequence": sequence,
                "question": question["text"] if question is not None else None,
                "answer": answer["text"] if answer is not None else None,
                "status": status,
                "reason": reason,
                "created_at": started_at,
                "completed_at": completed_at,
            },
        )
        for role in ("question", "answer"):
            message = exchange[role]
            if message is not None:
                turn_of_message.append((turn_id, message["id"]))
    return turn_of_message


def upgrade() -> None:
    """Replace the message log with ordered turns and one conversation per file."""
    op.create_table(
        "turns",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("conversation_id", sa.String(length=32), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("question", sa.String(length=8192), nullable=True),
        sa.Column("answer", sa.String(length=8192), nullable=True),
        sa.Column(
            "status", sa.String(length=16), nullable=False, server_default="pending"
        ),
        sa.Column("failure_reason", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["conversation_id"], ["conversations.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_turns_conversation_id", "turns", ["conversation_id"])
    op.create_index(
        "uq_turns_conversation_sequence",
        "turns",
        ["conversation_id", "sequence"],
        unique=True,
    )

    connection = op.get_bind()
    _merge_duplicate_conversations(connection)
    turn_of_message = _backfill_turns(connection)

    op.add_column("sources", sa.Column("turn_id", sa.String(length=32), nullable=True))
    for turn_id, message_id in turn_of_message:
        connection.execute(
            sa.text(
                "UPDATE sources SET turn_id = :turn_id WHERE message_id = :message_id"
            ),
            {"turn_id": turn_id, "message_id": message_id},
        )
    # SQLite cannot alter a column in place, so the rewrite runs as one batch.
    with op.batch_alter_table("sources") as batch:
        batch.alter_column(
            "turn_id", existing_type=sa.String(length=32), nullable=False
        )
        batch.create_foreign_key(
            "sources_turn_id_fkey", "turns", ["turn_id"], ["id"], ondelete="CASCADE"
        )
        batch.drop_column("message_id")
    op.drop_table("messages")
    op.create_index(
        "uq_conversations_file_id", "conversations", ["file_id"], unique=True
    )


def downgrade() -> None:
    """Restore the message log from the stored turns."""
    op.create_table(
        "messages",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("conversation_id", sa.String(length=32), nullable=False),
        sa.Column("sender", sa.String(length=16), nullable=False),
        sa.Column("text", sa.String(length=8192), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"], ["conversations.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.add_column(
        "sources", sa.Column("message_id", sa.String(length=32), nullable=True)
    )

    connection = op.get_bind()
    turns = (
        connection.execute(
            sa.text(
                "SELECT id, conversation_id, question, answer, created_at, completed_at "
                "FROM turns ORDER BY conversation_id, sequence"
            )
        )
        .mappings()
        .all()
    )
    for turn in turns:
        answer_message_id = None
        for sender, text, created_at in (
            ("user", turn["question"], turn["created_at"]),
            ("bot", turn["answer"], turn["completed_at"] or turn["created_at"]),
        ):
            if text is None:
                continue
            message_id = uuid.uuid4().hex
            connection.execute(
                sa.text(
                    "INSERT INTO messages (id, conversation_id, sender, text, created_at) "
                    "VALUES (:id, :conversation_id, :sender, :text, :created_at)"
                ),
                {
                    "id": message_id,
                    "conversation_id": turn["conversation_id"],
                    "sender": sender,
                    "text": text,
                    "created_at": created_at,
                },
            )
            if sender == "bot":
                answer_message_id = message_id
        if answer_message_id is not None:
            connection.execute(
                sa.text(
                    "UPDATE sources SET message_id = :message_id WHERE turn_id = :turn_id"
                ),
                {"message_id": answer_message_id, "turn_id": turn["id"]},
            )

    with op.batch_alter_table("sources") as batch:
        batch.alter_column(
            "message_id", existing_type=sa.String(length=32), nullable=False
        )
        batch.create_foreign_key(
            "sources_message_id_fkey",
            "messages",
            ["message_id"],
            ["id"],
            ondelete="CASCADE",
        )
        batch.drop_column("turn_id")
    op.drop_index("uq_conversations_file_id", table_name="conversations")
    op.drop_index("uq_turns_conversation_sequence", table_name="turns")
    op.drop_index("ix_turns_conversation_id", table_name="turns")
    op.drop_table("turns")
