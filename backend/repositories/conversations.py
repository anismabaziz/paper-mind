"""Persistence for conversations, messages, and citation sources."""

from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from db import Conversation, Message, Source
from repositories.base import BaseRepository


class ConversationRepository(BaseRepository):
    """Read and update the conversation aggregate."""

    def create_conversation(self, file_id: str) -> str:
        """Create a conversation for one stored document."""
        with self._session_factory() as session, session.begin():
            conversation = Conversation(file_id=file_id)
            session.add(conversation)
            session.flush()
            return conversation.id

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

    def add_message(
        self,
        conversation_id: str,
        sender: str,
        text: str,
        sources: list[dict[str, Any]] | None = None,
    ) -> str:
        """Persist one message and its ordered citation sources."""
        with self._session_factory() as session, session.begin():
            message = Message(
                conversation_id=conversation_id,
                sender=sender,
                text=text,
            )
            session.add(message)
            session.flush()
            for source in sources or []:
                session.add(
                    Source(
                        message_id=message.id,
                        content=source["content"],
                        document=source["document"],
                        chunk_index=source["chunk_index"],
                        score=source["score"],
                        page=source.get("page", source.get("page_no")),
                    )
                )
            return message.id

    def get_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        """Return one conversation's messages in persisted order."""
        with self._session_factory() as session:
            rows = session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at)
            ).all()
            return [
                {
                    "id": message.id,
                    "text": message.text,
                    "sender": message.sender,
                    "sources": [
                        self._source_dict(source)
                        for source in self._sources_for(session, message.id)
                    ],
                    "created_at": (
                        message.created_at.isoformat() if message.created_at else None
                    ),
                }
                for message in rows
            ]

    def delete_messages(self, conversation_id: str) -> None:
        """Delete every message in one conversation."""
        with self._session_factory() as session, session.begin():
            session.execute(
                delete(Message).where(Message.conversation_id == conversation_id)
            )

    @staticmethod
    def _sources_for(session: Session, message_id: str) -> list[Source]:
        return list(
            session.scalars(
                select(Source)
                .where(Source.message_id == message_id)
                .order_by(Source.score.desc(), Source.chunk_index)
            ).all()
        )

    @staticmethod
    def _source_dict(source: Source) -> dict[str, Any]:
        return {
            "content": source.content,
            "document": source.document,
            "chunk_index": source.chunk_index,
            "score": source.score,
            "page": source.page,
        }
