import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from uuid import UUID, uuid4

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, Header, HTTPException
from pgvector.sqlalchemy import Vector
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import (
    DateTime,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://margin:margin@localhost:5432/margin")
EMBEDDING_URL = "https://integrate.api.nvidia.com/v1/embeddings"
EMBEDDING_MODEL = "nvidia/nemotron-3.5-embed-1b"
EMBEDDING_DIMENSIONS = 2048
RELATED_THOUGHT_THRESHOLD = 0.60
logger = logging.getLogger(__name__)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
router = APIRouter(prefix="/thoughts", tags=["thoughts"])
_schema_lock = Lock()
_schema_ready = False


class Base(DeclarativeBase):
    pass


class ThoughtRecord(Base):
    __tablename__ = "thoughts"
    __table_args__ = (UniqueConstraint("client_id", "source_key"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    client_id: Mapped[UUID] = mapped_column(index=True)
    source_key: Mapped[str] = mapped_column(String(300))
    note: Mapped[str] = mapped_column(Text)
    paragraph_text: Mapped[str] = mapped_column(Text)
    paragraph_index: Mapped[int] = mapped_column()
    article_title: Mapped[str] = mapped_column(String(300))
    article_url: Mapped[str] = mapped_column(String(2048))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS))


class ThoughtDismissal(Base):
    __tablename__ = "thought_dismissals"
    __table_args__ = (UniqueConstraint("client_id", "source_key", "thought_id"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    client_id: Mapped[UUID] = mapped_column(index=True)
    source_key: Mapped[str] = mapped_column(String(300))
    thought_id: Mapped[UUID] = mapped_column(ForeignKey("thoughts.id", ondelete="CASCADE"))


class ThoughtInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    client_id: UUID = Field(alias="clientId")
    source_key: str = Field(alias="sourceKey", min_length=1, max_length=300)
    note: str = Field(max_length=10_000)
    paragraph_text: str = Field(alias="paragraphText", min_length=1, max_length=10_000)
    paragraph_index: int = Field(alias="paragraphIndex", ge=0)
    article_title: str = Field(alias="articleTitle", min_length=1, max_length=300)
    article_url: str = Field(alias="articleUrl", max_length=2048)
    created_at: datetime | None = Field(default=None, alias="createdAt")


class RelatedThoughtQuery(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    client_id: UUID = Field(alias="clientId")
    source_key: str = Field(alias="sourceKey", min_length=1, max_length=300)
    paragraph_text: str = Field(alias="paragraphText", min_length=1, max_length=10_000)
    note: str = Field(default="", max_length=10_000)


class DismissThoughtInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    client_id: UUID = Field(alias="clientId")
    source_key: str = Field(alias="sourceKey", min_length=1, max_length=300)
    thought_id: UUID = Field(alias="thoughtId")


def ensure_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        try:
            with engine.begin() as connection:
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                Base.metadata.create_all(connection)
        except Exception as error:
            raise HTTPException(
                status_code=503,
                detail="The memory database is unavailable. Start PostgreSQL and try again.",
            ) from error
        _schema_ready = True


def require_api_key(authorization: str | None) -> str:
    api_key = authorization.removeprefix("Bearer ").strip() if authorization else ""
    if not api_key:
        raise HTTPException(status_code=401, detail="Add an NVIDIA API key in Settings.")
    return api_key


async def embed_text(text_value: str, api_key: str, input_type: str) -> list[float]:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                EMBEDDING_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "input": [text_value],
                    "model": EMBEDDING_MODEL,
                    "input_type": input_type,
                },
            )
            response.raise_for_status()
            embedding = response.json()["data"][0]["embedding"]
    except httpx.HTTPStatusError as error:
        status_code = error.response.status_code
        logger.warning(
            "NVIDIA embedding request failed: status=%s model=%s response=%s",
            status_code,
            EMBEDDING_MODEL,
            error.response.text[:500],
        )
        if status_code in (401, 403):
            detail = "NVIDIA rejected the API key. Check the key in Settings."
        elif status_code == 404:
            detail = "The NVIDIA embedding model is unavailable. Update the app and retry."
        elif status_code == 429:
            detail = "NVIDIA rate limited the embedding request. Try again shortly."
        else:
            detail = "NVIDIA could not create a memory connection right now."
        raise HTTPException(status_code=502, detail=detail) from error
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
        logger.warning("NVIDIA embedding response could not be used: %s", error)
        raise HTTPException(
            status_code=502,
            detail="NVIDIA could not create a memory connection right now.",
        ) from error

    if not isinstance(embedding, list) or len(embedding) != EMBEDDING_DIMENSIONS:
        raise HTTPException(
            status_code=502,
            detail="NVIDIA returned an embedding with an unexpected size.",
        )
    return [float(value) for value in embedding]


def serialize_thought(record: ThoughtRecord, score: float) -> dict[str, object]:
    return {
        "id": str(record.id),
        "note": record.note,
        "sourceKey": record.source_key,
        "paragraphIndex": record.paragraph_index,
        "articleTitle": record.article_title,
        "articleUrl": record.article_url,
        "createdAt": record.created_at.isoformat(),
        "score": score,
    }


@router.put("/{source_key:path}")
async def save_thought(
    source_key: str,
    thought: ThoughtInput,
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    if source_key != thought.source_key:
        raise HTTPException(status_code=422, detail="Thought source did not match the URL.")
    ensure_schema()

    with Session(engine) as session:
        existing = session.scalar(
            select(ThoughtRecord).where(
                ThoughtRecord.client_id == thought.client_id,
                ThoughtRecord.source_key == source_key,
            )
        )
        if not thought.note.strip():
            if existing:
                session.execute(delete(ThoughtRecord).where(ThoughtRecord.id == existing.id))
                session.commit()
            return {"saved": False}
        if existing and (
            existing.note == thought.note
            and existing.paragraph_text == thought.paragraph_text
            and existing.article_title == thought.article_title
            and existing.article_url == thought.article_url
            and existing.paragraph_index == thought.paragraph_index
        ):
            return {"saved": True, "id": str(existing.id)}

    api_key = require_api_key(authorization)
    embedding = await embed_text(
        f"Thought: {thought.note.strip()}\n\nPassage: {thought.paragraph_text.strip()}",
        api_key,
        "passage",
    )
    record_id = existing.id if existing else uuid4()
    statement = (
        insert(ThoughtRecord)
        .values(
            id=record_id,
            client_id=thought.client_id,
            source_key=source_key,
            note=thought.note.strip(),
            paragraph_text=thought.paragraph_text.strip(),
            paragraph_index=thought.paragraph_index,
            article_title=thought.article_title.strip(),
            article_url=thought.article_url,
            created_at=thought.created_at or datetime.now(UTC),
            embedding=embedding,
        )
        .on_conflict_do_update(
            index_elements=["client_id", "source_key"],
            set_={
                "note": thought.note.strip(),
                "paragraph_text": thought.paragraph_text.strip(),
                "paragraph_index": thought.paragraph_index,
                "article_title": thought.article_title.strip(),
                "article_url": thought.article_url,
                "embedding": embedding,
                "created_at": thought.created_at or datetime.now(UTC),
            },
        )
        .returning(ThoughtRecord.id)
    )
    try:
        with Session(engine) as session:
            if existing:
                session.execute(
                    delete(ThoughtDismissal).where(ThoughtDismissal.thought_id == existing.id)
                )
            saved_id = session.scalar(statement)
            session.commit()
    except Exception as error:
        raise HTTPException(status_code=503, detail="Could not save this memory.") from error
    return {"saved": True, "id": str(saved_id)}


@router.post("/related")
async def find_related_thought(
    query: RelatedThoughtQuery,
    authorization: str | None = Header(default=None),
) -> dict[str, object]:
    ensure_schema()
    # Reuse the saved vector for note-to-note comparisons. Cosine similarity
    # between the same two stored vectors is identical in both directions.
    saved_embedding = None
    if query.note.strip():
        with Session(engine) as session:
            current = session.scalar(
                select(ThoughtRecord).where(
                    ThoughtRecord.client_id == query.client_id,
                    ThoughtRecord.source_key == query.source_key,
                    ThoughtRecord.note == query.note.strip(),
                    ThoughtRecord.paragraph_text == query.paragraph_text.strip(),
                )
            )
            if current is not None:
                saved_embedding = list(current.embedding)
    if saved_embedding is not None:
        embedding = saved_embedding
    else:
        api_key = require_api_key(authorization)
        query_text = (
            f"Thought: {query.note.strip()}\n\nPassage: {query.paragraph_text.strip()}"
            if query.note.strip()
            else query.paragraph_text.strip()
        )
        embedding = await embed_text(
            query_text, api_key, "passage" if query.note.strip() else "query"
        )
    distance = ThoughtRecord.embedding.cosine_distance(embedding).label("distance")
    dismissed = select(ThoughtDismissal.id).where(
        ThoughtDismissal.client_id == query.client_id,
        ThoughtDismissal.source_key == query.source_key,
        ThoughtDismissal.thought_id == ThoughtRecord.id,
    )
    try:
        with Session(engine) as session:
            matches = session.execute(
                select(ThoughtRecord, distance)
                .where(
                    ThoughtRecord.client_id == query.client_id,
                    ThoughtRecord.source_key != query.source_key,
                    ~dismissed.exists(),
                )
                .order_by(distance)
                .limit(5)
            ).all()
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="Could not search your margins right now.",
        ) from error

    logger.warning(
        "Memory search: candidates=%s top_score=%s threshold=%.2f has_note=%s",
        len(matches),
        round(1 - float(matches[0][1]), 4) if matches else None,
        RELATED_THOUGHT_THRESHOLD,
        bool(query.note.strip()),
    )
    for record, cosine_distance in matches:
        score = 1 - float(cosine_distance)
        if score >= RELATED_THOUGHT_THRESHOLD:
            return {"thought": serialize_thought(record, score)}
    return {"thought": None}


@router.post("/dismiss")
def dismiss_related_thought(dismissal: DismissThoughtInput) -> dict[str, bool]:
    ensure_schema()
    statement = (
        insert(ThoughtDismissal)
        .values(
            id=uuid4(),
            client_id=dismissal.client_id,
            source_key=dismissal.source_key,
            thought_id=dismissal.thought_id,
        )
        .on_conflict_do_nothing(index_elements=["client_id", "source_key", "thought_id"])
    )
    try:
        with Session(engine) as session:
            session.execute(statement)
            session.commit()
    except Exception as error:
        raise HTTPException(status_code=503, detail="Could not save that preference.") from error
    return {"dismissed": True}
