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