"""Persisted Ask Margin conversations, scoped to a reader (clientId) and article."""

import re
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import DateTime, Index, String, delete, select
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.orm import Mapped, Session, mapped_column
from starlette.concurrency import run_in_threadpool

from memory import Base, engine, ensure_schema

router = APIRouter(prefix="/companion/chats", tags=["companion"])
MAX_CHATS_PER_ARTICLE = 30
CHAT_ID_PATTERN = r"^[A-Za-z0-9._-]{1,64}$"


class ChatRecord(Base):
    __tablename__ = "companion_chats"
    __table_args__ = (Index("ix_companion_chats_owner_article", "client_id", "article_url"),)

    client_id: Mapped[UUID] = mapped_column(primary_key=True)
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    article_url: Mapped[str] = mapped_column(String(2048))
    title: Mapped[str] = mapped_column(String(300))
    turns: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ChatSource(BaseModel):
    id: int
    kind: str = Field(max_length=40)
    title: str = Field(max_length=300)
    url: str = Field(max_length=2048)
    paragraph_index: int = Field(alias="paragraphIndex", ge=0)
    text: str = Field(max_length=20_000)

    model_config = ConfigDict(populate_by_name=True)


class ChatTurn(BaseModel):
    id: float
    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(default="", max_length=20_000)
    sources: list[ChatSource] = Field(default_factory=list, max_length=24)
    status: Literal["streaming", "done", "stopped", "error"]
    error: str | None = Field(default=None, max_length=500)


class ChatInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    client_id: UUID = Field(alias="clientId")
    article_url: str = Field(alias="articleUrl", min_length=1, max_length=2048)
    title: str = Field(min_length=1, max_length=300)
    turns: list[ChatTurn] = Field(min_length=1, max_length=200)
    updated_at: datetime = Field(alias="updatedAt")


def serialize(record: ChatRecord) -> dict[str, object]:
    return {
        "id": record.id,
        "title": record.title,
        "updatedAt": int(record.updated_at.timestamp() * 1000),
        "turns": record.turns,
    }


def list_chats_sync(client_id: UUID, article_url: str) -> list[dict[str, object]]:
    ensure_schema()
    with Session(engine) as session:
        records = session.scalars(
            select(ChatRecord)
            .where(ChatRecord.client_id == client_id, ChatRecord.article_url == article_url)
            .order_by(ChatRecord.updated_at.desc())
            .limit(MAX_CHATS_PER_ARTICLE)
        ).all()
        return [serialize(record) for record in records]


def save_chat_sync(chat_id: str, chat: ChatInput) -> None:
    ensure_schema()
    # An answer still streaming when saved is incomplete; store it as stopped.
    turns = [
        {
            **turn.model_dump(by_alias=True),
            "status": "stopped" if turn.status == "streaming" else turn.status,
        }
        for turn in chat.turns
    ]
    updated_at = chat.updated_at if chat.updated_at.tzinfo else chat.updated_at.replace(tzinfo=UTC)
    values = {
        "client_id": chat.client_id,
        "id": chat_id,
        "article_url": chat.article_url,
        "title": chat.title[:300],
        "turns": turns,
        "updated_at": updated_at,
    }
    with Session(engine) as session:
        session.execute(
            insert(ChatRecord)
            .values(**values)
            .on_conflict_do_update(
                index_elements=["client_id", "id"],
                set_={key: values[key] for key in ("article_url", "title", "turns", "updated_at")},
            )
        )
        stale = (
            select(ChatRecord.id)
            .where(
                ChatRecord.client_id == chat.client_id,
                ChatRecord.article_url == chat.article_url,
            )
            .order_by(ChatRecord.updated_at.desc())
            .offset(MAX_CHATS_PER_ARTICLE)
        )
        session.execute(
            delete(ChatRecord).where(
                ChatRecord.client_id == chat.client_id, ChatRecord.id.in_(stale.scalar_subquery())
            )
        )
        session.commit()


def delete_chat_sync(client_id: UUID, chat_id: str) -> None:
    ensure_schema()
    with Session(engine) as session:
        session.execute(
            delete(ChatRecord).where(ChatRecord.client_id == client_id, ChatRecord.id == chat_id)
        )
        session.commit()


@router.get("")
async def list_chats(
    client_id: UUID = Query(alias="clientId"),
    article_url: str = Query(alias="articleUrl", min_length=1, max_length=2048),
) -> dict[str, object]:
    return {"chats": await run_in_threadpool(list_chats_sync, client_id, article_url)}


@router.put("/{chat_id}")
async def save_chat(chat_id: str, chat: ChatInput) -> dict[str, bool]:
    if not re.fullmatch(CHAT_ID_PATTERN, chat_id):
        raise HTTPException(status_code=422, detail="Invalid chat id.")
    await run_in_threadpool(save_chat_sync, chat_id, chat)
    return {"saved": True}


@router.delete("/{chat_id}")
async def delete_chat(chat_id: str, client_id: UUID = Query(alias="clientId")) -> dict[str, bool]:
    await run_in_threadpool(delete_chat_sync, client_id, chat_id)
    return {"deleted": True}
