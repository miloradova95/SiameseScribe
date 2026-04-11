# services/main_backend/db/models.py
# SQLite table definitions using SQLModel.
# Only the main backend uses this — the ML backend never touches the DB.

from sqlmodel import SQLModel, Field, Column
from sqlalchemy import JSON
from typing import Optional
from datetime import datetime, timezone


class Image(SQLModel, table=True):
    __tablename__ = "images"

    id: Optional[int] = Field(default=None, primary_key=True)
    file_path: str = Field(..., description="Absolute path: /data/images/{id}.png")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Patch(SQLModel, table=True):
    __tablename__ = "patches"

    id: Optional[int] = Field(default=None, primary_key=True)
    image_id: int = Field(..., foreign_key="images.id", index=True)
    patch_path: str = Field(..., description="Absolute path: /data/patches/{id}.png")
    bbox: dict = Field(sa_column=Column(JSON), description='{"x","y","width","height"}')
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Embedding(SQLModel, table=True):
    __tablename__ = "embeddings"

    id: Optional[int] = Field(default=None, primary_key=True)
    patch_id: int = Field(..., foreign_key="patches.id", unique=True, index=True)
    vector: list[float] = Field(sa_column=Column(JSON), description="32-dim float list")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Feedback(SQLModel, table=True):
    __tablename__ = "feedback"

    id: Optional[int] = Field(default=None, primary_key=True)
    query_patch_id: int = Field(..., foreign_key="patches.id", index=True)
    result_patch_id: int = Field(..., foreign_key="patches.id", index=True)
    label: int = Field(..., description="1 = similar | 0 = not_similar")
    used_for_retrain: bool = Field(default=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))