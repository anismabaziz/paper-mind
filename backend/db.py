"""Database engine and ORM models."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    sessionmaker,
)
from sqlalchemy.pool import StaticPool

from settings import get_settings


def _new_id():
    return uuid.uuid4().hex


APP_SETTINGS_ID = "app"


class Base(DeclarativeBase):
    """Base for all persisted models."""


class FileRecord(Base):
    """Metadata for one stored document."""

    __tablename__ = "files"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    filename: Mapped[str] = mapped_column(String(255), unique=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    original_filename: Mapped[str | None] = mapped_column(
        String(255), nullable=True, default=None
    )
    is_processed: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class AppSettings(Base):
    """Global application settings."""

    __tablename__ = "app_settings"

    id: Mapped[str] = mapped_column(
        String(32), primary_key=True, default=lambda: APP_SETTINGS_ID
    )
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(128))
    encrypted_api_key: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Conversation(Base):
    """Conversation tied to one stored document."""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    file_id: Mapped[str] = mapped_column(ForeignKey("files.id", ondelete="CASCADE"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Message(Base):
    """Message in a document conversation."""

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
    """Citation source attached to a message."""

    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_id)
    message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE")
    )
    content: Mapped[str] = mapped_column(String(8192))
    document: Mapped[str] = mapped_column(String(255))
    chunk_index: Mapped[int] = mapped_column()
    score: Mapped[float] = mapped_column()
    page: Mapped[int | None] = mapped_column(Integer, default=None, nullable=True)


def _engine_kwargs(url: str):
    if url.startswith("sqlite"):
        kwargs: dict[str, Any] = {"connect_args": {"check_same_thread": False}}
        if ":memory:" in url:
            kwargs["poolclass"] = StaticPool
        return kwargs
    return {}


_engine = None
SessionLocal = None


def get_session_factory():
    """Build the engine and session factory on first use."""
    global _engine, SessionLocal
    if _engine is None:
        database_url = get_settings().database.database_url
        _engine = create_engine(database_url, **_engine_kwargs(database_url))
        SessionLocal = sessionmaker(bind=_engine)
    return SessionLocal
