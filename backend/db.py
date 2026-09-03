"""Database engine, ORM models, and the repository seam.

All relational access goes through :class:`Repository`; routes and services
never touch a Session directly.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, String, create_engine, func, select
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    sessionmaker,
)
from sqlalchemy.pool import StaticPool

import config


def _new_id():
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class FileRecord(Base):
    __tablename__ = "files"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    filename: Mapped[str] = mapped_column(String(255), unique=True)
    is_processed: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    file_id: Mapped[str] = mapped_column(ForeignKey("files.id", ondelete="CASCADE"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    sender: Mapped[str] = mapped_column(String(16))
    text: Mapped[str] = mapped_column(String(8192))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE")
    )
    content: Mapped[str] = mapped_column(String(8192))
    document: Mapped[str] = mapped_column(String(255))
    chunk_index: Mapped[int] = mapped_column()
    score: Mapped[float] = mapped_column()


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    password_hash: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


def _engine_kwargs(url: str):
    # Tests may use sqlite; only the in-memory variant needs special care.
    if url.startswith("sqlite"):
        kwargs = {"connect_args": {"check_same_thread": False}}
        if ":memory:" in url:
            # Every session must share one connection or the schema
            # disappears between sessions.
            kwargs["poolclass"] = StaticPool
        return kwargs
    return {}


_engine = None
SessionLocal = None


def get_session_factory():
    """Build the engine and session factory on first use, from the live env."""
    global _engine, SessionLocal
    if _engine is None:
        _engine = create_engine(config.DATABASE_URL, **_engine_kwargs(config.DATABASE_URL))
        SessionLocal = sessionmaker(bind=_engine)
    return SessionLocal


def _to_dict(record, **extra):
    created = record.created_at or datetime.now(timezone.utc)
    data = {"id": record.id, "created_at": created.isoformat()}
    data.update(extra)
    return data


class Repository:
    """Repository seam: the only relational access point for the app."""

    def __init__(self, session_factory=None):
        # Resolved lazily so importing this module never requires env vars.
        self._factory_override = session_factory
        self._resolved = None

    @property
    def _session_factory(self):
        if self._factory_override is not None:
            return self._factory_override
        if self._resolved is None:
            self._resolved = get_session_factory()
        return self._resolved

    # -- files ----------------------------------------------------------

    def create_file(self, filename: str) -> dict:
        with self._session_factory() as session, session.begin():
            record = FileRecord(filename=filename)
            session.add(record)
            session.flush()
            return self._file_dict(record)

    def get_file(self, filename: str) -> dict | None:
        with self._session_factory() as session:
            record = session.scalars(
                select(FileRecord).where(FileRecord.filename == filename)
            ).first()
            return self._file_dict(record) if record else None

    def list_files(self) -> list[dict]:
        with self._session_factory() as session:
            records = session.scalars(
                select(FileRecord).order_by(FileRecord.created_at)
            ).all()
            return [self._file_dict(r) for r in records]

    def set_processed(self, filename: str, status: bool = True) -> None:
        with self._session_factory() as session, session.begin():
            record = session.scalars(
                select(FileRecord).where(FileRecord.filename == filename)
            ).first()
            if record:
                record.is_processed = status

    def delete_file(self, file_id: str) -> None:
        with self._session_factory() as session, session.begin():
            record = session.get(FileRecord, file_id)
            if record:
                session.delete(record)

    @staticmethod
    def _file_dict(record):
        return _to_dict(record, filename=record.filename, is_processed=record.is_processed)

    # -- users ------------------------------------------------------------

    def create_user(self, email: str, password_hash: str) -> dict:
        with self._session_factory() as session, session.begin():
            user = User(email=email, password_hash=password_hash)
            session.add(user)
            session.flush()
            return {"id": user.id, "email": user.email}

    def get_user_by_email(self, email: str) -> dict | None:
        with self._session_factory() as session:
            user = session.scalars(
                select(User).where(User.email == email)
            ).first()
            if not user:
                return None
            return {"id": user.id, "email": user.email, "password_hash": user.password_hash}

    # -- conversations ----------------------------------------------------

    def create_conversation(self, file_id: str) -> str:
        with self._session_factory() as session, session.begin():
            conversation = Conversation(file_id=file_id)
            session.add(conversation)
            session.flush()
            return conversation.id

    def get_conversation_id(self, file_id: str) -> str | None:
        with self._session_factory() as session:
            return session.scalars(
                select(Conversation.id).where(Conversation.file_id == file_id)
            ).first()

    def delete_conversation(self, conversation_id: str) -> None:
        with self._session_factory() as session, session.begin():
            conversation = session.get(Conversation, conversation_id)
            if conversation:
                session.delete(conversation)

    # -- messages ---------------------------------------------------------

    def add_message(
        self, conversation_id: str, sender: str, text: str, sources=None
    ) -> str:
        with self._session_factory() as session, session.begin():
            message = Message(
                conversation_id=conversation_id, sender=sender, text=text
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
                    )
                )
            return message.id

    def get_messages(self, conversation_id: str) -> list[dict]:
        with self._session_factory() as session:
            rows = session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at)
            ).all()
            return [
                {
                    "id": m.id,
                    "text": m.text,
                    "sender": m.sender,
                    "sources": [
                        self._source_dict(s) for s in self._sources_for(session, m.id)
                    ],
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                }
                for m in rows
            ]

    @staticmethod
    def _sources_for(session, message_id: str) -> list[Source]:
        # Score order mirrors the retrieval order the LLM received, so the
        # Sources panel shows the same ranking for stored and fresh answers.
        return list(
            session.scalars(
                select(Source)
                .where(Source.message_id == message_id)
                .order_by(Source.score.desc(), Source.chunk_index)
            ).all()
        )

    @staticmethod
    def _source_dict(source: Source) -> dict:
        return {
            "content": source.content,
            "document": source.document,
            "chunk_index": source.chunk_index,
            "score": source.score,
        }

    def delete_messages(self, conversation_id: str) -> None:
        with self._session_factory() as session, session.begin():
            session.query(Message).where(
                Message.conversation_id == conversation_id
            ).delete(synchronize_session=False)


repository = Repository()
