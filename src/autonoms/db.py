"""Database setup: the engine, the session dependency, and the tasks table."""

from collections.abc import Iterator
from datetime import datetime
from functools import lru_cache

from sqlalchemy import JSON, DateTime, Engine, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from autonoms.config import get_settings


class Base(DeclarativeBase):
    pass


class TaskRow(Base):
    """Python view of the `tasks` table. One row per task."""

    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    owner: Mapped[str] = mapped_column(String(100))
    repo: Mapped[str] = mapped_column(String(100))
    issue_number: Mapped[int] = mapped_column(Integer)
    issue_title: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String(20))
    # A snapshot of the issue as it was when the task started (JSON column).
    issue: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class MemoryRow(Base):
    """Long-term memory: short notes that outlive a single task."""

    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    repo: Mapped[str] = mapped_column(String(201), index=True)  # "owner/name", or "*" = applies everywhere
    kind: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    source_task_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    use_count: Mapped[int] = mapped_column(Integer, default=0)


@lru_cache
def get_engine() -> Engine:
    # pool_pre_ping checks a pooled connection is alive before using it.
    return create_engine(get_settings().database_url, pool_pre_ping=True)


def init_db() -> None:
    """Create missing tables. Existing tables and rows are never touched."""
    Base.metadata.create_all(get_engine())


def get_session() -> Iterator[Session]:
    with Session(get_engine()) as session:
        yield session
