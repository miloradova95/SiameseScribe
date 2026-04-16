# services/main_backend/db/models.py
# SQLite table definitions using SQLModel.
# Only the main backend uses this — the ML backend never touches the DB.

from sqlmodel import SQLModel, Field, Column
from typing import Optional


class Image(SQLModel, table=True):
    __tablename__ = "images"

    id: Optional[int] = Field(default=None, primary_key=True)
    fileName: str = Field(..., description="Original file name of the uploaded image")
    filePath: str = Field(..., description="Absolute path: /data/dataset/preprocessed/")
    group: Optional[str] = Field(default=None, description="Optional group identifier for categorization")
