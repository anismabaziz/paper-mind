"""Persistence for conversations, ordered turns, and citation sources."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from db import (
    TURN_ANSWERED,
    TURN_CANCELLED,
    TURN_FAILED,
    TURN_PENDING,
    TERMINAL_TURN_STATES,
    Conversation,
    Source,
    Turn,
)
from repositories.base import BaseRepository

# Two callers can pick the same next sequence when they read the maximum at
# the same time. The unique index rejects the loser, who reads again.
_SEQUENCE_ATTEMPTS = 5


def _delete_conversation_rows(session: Session, conversation_id: str) -> None:
    """Delete one conversation's turns and their citation sources."""
    turn_ids = list(
        session.scalars(
            select(Turn.id).where(Turn.conversation_id == conversation_id)
        ).all()
    )
    if turn_ids:
        session.execute(delete(Source).where(Source.turn_id.in_(turn_ids)))
    session.execute(delete(Turn).where(Turn.conversation_id == conversation_id))


def _lock(session: Session, statement) -> None:
    """Take a row lock on Postgres, which has nothing to lock on SQLite."""
    if session.get_bind().dialect.name == "postgresql":
        session.execute(statement.with_for_update())


def _is_sequence_conflict(error: IntegrityError) -> bool:
    """Report whether an insert lost the race for a turn sequence."""
    diagnostic = getattr(getattr(error, "orig", None), "diag", None)
    constraint = getattr(diagnostic, "constraint_name", "") or ""
    if "turns_conversation_sequence" in constraint:
        return True
    return "turns.conversation_id, turns.sequence" in str(error)


def _next_sequence(session: Session, conversation_id: str) -> int:
    """Return the sequence a new turn in this conversation takes."""
    # Holding the conversation row for the rest of the transaction makes two
    # concurrent starts take their sequences one after the other.
    _lock(session, select(Conversation.id).where(Conversation.id == conversation_id))
    highest = session.scalar(
        select(func.coalesce(func.max(Turn.sequence), 0)).where(
            Turn.conversation_id == conversation_id
        )
    )
    return int(highest or 0) + 1


class ConversationRepository(BaseRepository):
    """Read and update the conversation aggregate."""

    def ensure_conversation(self, file_id: str) -> str:
        """Return the conversation for a document, creating it once if absent."""
        existing = self.get_conversation_id(file_id)
        if existing:
            return existing
        try:
            with self._session_factory() as session, session.begin():
                conversation = Conversation(file_id=file_id)
                session.add(conversation)
                session.flush()
                return conversation.id
        except IntegrityError:
            # Another caller created it between the read and the insert.
            existing = self.get_conversation_id(file_id)
            if existing:
                return existing
            raise

    def get_conversation_id(self, file_id: str) -> str | None:
        """Return the conversation identifier for a document, if present."""
        with self._session_factory() as session:
            return session.scalars(
                select(Conversation.id).where(Conversation.file_id == file_id)
            ).first()

    def delete_conversation(self, conversation_id: str) -> None:
        """Delete one conversation."""
        with self._session_factory() as session, session.begin():
            conversation = session.get(Conversation, conversation_id)
            if conversation:
                session.delete(conversation)

    def start_turn(self, conversation_id: str, question: str) -> str:
        """Commit a question as a pending turn before the provider is called."""
        for attempt in range(_SEQUENCE_ATTEMPTS):
            try:
                with self._session_factory() as session, session.begin():
                    turn = Turn(
                        conversation_id=conversation_id,
                        sequence=_next_sequence(session, conversation_id),
                        question=question,
                        status=TURN_PENDING,
                    )
                    session.add(turn)
                    session.flush()
                    return turn.id
            except IntegrityError as error:
                if (
                    not _is_sequence_conflict(error)
                    or attempt == _SEQUENCE_ATTEMPTS - 1
                ):
                    raise
        raise RuntimeError("could not allocate a turn sequence")

    def complete_turn(
        self,
        turn_id: str,
        answer: str,
        sources: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Record an answer and its ordered citation sources on a pending turn."""
        return self._finish_turn(turn_id, TURN_ANSWERED, answer, sources, None)

    def fail_turn(self, turn_id: str, message: str, reason: str) -> bool:
        """Record a terminal failure with the reply the user was shown."""
        return self._finish_turn(turn_id, TURN_FAILED, message, None, reason)

    def cancel_turn(self, turn_id: str, reason: str) -> bool:
        """Record a terminal cancellation for a question the user walked away from."""
        return self._finish_turn(turn_id, TURN_CANCELLED, None, None, reason)

    def _finish_turn(
        self,
        turn_id: str,
        status: str,
        answer: str | None,
        sources: list[dict[str, Any]] | None,
        failure_reason: str | None,
    ) -> bool:
        """Close one pending turn, leaving a turn that already ended untouched."""
        with self._session_factory() as session, session.begin():
            # Two outcomes can reach one turn at once; the lock makes the
            # second one see the state the first one wrote.
            _lock(session, select(Turn).where(Turn.id == turn_id))
            turn = session.get(Turn, turn_id)
            if turn is None or turn.status in TERMINAL_TURN_STATES:
                return False
            turn.status = status
            turn.answer = answer
            turn.failure_reason = failure_reason
            turn.completed_at = datetime.now(timezone.utc)
            for source in sources or []:
                session.add(
                    Source(
                        turn_id=turn.id,
                        content=source["content"],
                        document=source["document"],
                        chunk_index=source["chunk_index"],
                        score=source["score"],
                        page=source.get("page", source.get("page_no")),
                    )
                )
            return True

    def get_turns(self, conversation_id: str) -> list[dict[str, Any]]:
        """Return one conversation's turns in sequence order with their sources."""
        with self._session_factory() as session:
            rows = session.scalars(
                select(Turn)
                .where(Turn.conversation_id == conversation_id)
                .order_by(Turn.sequence)
            ).all()
            return [
                {
                    "id": turn.id,
                    "sequence": turn.sequence,
                    "question": turn.question,
                    "answer": turn.answer,
                    "status": turn.status,
                    "failure_reason": turn.failure_reason,
                    "started_at": _isoformat(turn.created_at),
                    "completed_at": _isoformat(turn.completed_at),
                    "sources": [
                        self._source_dict(source)
                        for source in self._sources_for(session, turn.id)
                    ],
                }
                for turn in rows
            ]

    def get_recent_turns(
        self,
        conversation_id: str,
        limit: int,
        statuses: tuple[str, ...] = (TURN_ANSWERED,),
    ) -> list[dict[str, Any]]:
        """
        Return the newest turns that carry an answer, oldest first.

        Only completed exchanges are eligible: a pending, failed, or cancelled
        turn has nothing to refer back to, and letting one into the window
        would hand the model a question the user never got answered.
        """
        if limit <= 0 or not statuses:
            return []
        with self._session_factory() as session:
            rows = list(
                session.scalars(
                    select(Turn)
                    .where(
                        Turn.conversation_id == conversation_id,
                        Turn.status.in_(statuses),
                    )
                    .order_by(Turn.sequence.desc())
                    .limit(limit)
                ).all()
            )
            return [
                {
                    "id": turn.id,
                    "sequence": turn.sequence,
                    "question": turn.question,
                    "answer": turn.answer,
                    "status": turn.status,
                }
                for turn in reversed(rows)
            ]

    def count_answered_turns(self, conversation_id: str) -> int:
        """Return how many Turns in the Conversation carry an answer."""
        with self._session_factory() as session:
            return int(
                session.scalar(
                    select(func.count(Turn.id)).where(
                        Turn.conversation_id == conversation_id,
                        Turn.status == TURN_ANSWERED,
                    )
                )
                or 0
            )

    def get_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        """Return the conversation's exchanges as an ordered message list."""
        messages: list[dict[str, Any]] = []
        for turn in self.get_turns(conversation_id):
            for sender, text, sources, created_at in (
                ("user", turn["question"], [], turn["started_at"]),
                ("bot", turn["answer"], turn["sources"], turn["completed_at"]),
            ):
                if text is None:
                    continue
                messages.append(
                    {
                        "id": f"{turn['id']}-{sender}",
                        "text": text,
                        "sender": sender,
                        "sources": sources,
                        "created_at": created_at,
                        "turn_id": turn["id"],
                        "turn_sequence": turn["sequence"],
                        "turn_status": turn["status"],
                    }
                )
        return messages

    def delete_conversation_tree(self, conversation_id: str) -> None:
        """Delete a conversation with its turns and citation sources."""
        with self._session_factory() as session, session.begin():
            _delete_conversation_rows(session, conversation_id)
            conversation = session.get(Conversation, conversation_id)
            if conversation:
                session.delete(conversation)

    @staticmethod
    def _sources_for(session: Session, turn_id: str) -> list[Source]:
        """Return one turn's citation sources in display order."""
        return list(
            session.scalars(
                select(Source)
                .where(Source.turn_id == turn_id)
                .order_by(Source.score.desc(), Source.chunk_index)
            ).all()
        )

    @staticmethod
    def _source_dict(source: Source) -> dict[str, Any]:
        """Return one citation source as the shape the API serves."""
        return {
            "content": source.content,
            "document": source.document,
            "chunk_index": source.chunk_index,
            "score": source.score,
            "page": source.page,
        }


def _isoformat(value: datetime | None) -> str | None:
    """Return a stored timestamp as an ISO 8601 string, or None when unset."""
    return value.isoformat() if value else None
