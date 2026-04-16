from sqlmodel import SQLModel, Field, Column
from sqlalchemy import JSON
from sqlmodel import SQLModel
from typing import Optional
from datetime import datetime, timezone

class Patch(SQLModel, table=True):
    __tablename__ = "patches"

    id: Optional[int] = Field(default=None, primary_key=True)
    image_id: int = Field(..., foreign_key="images.id", index=True)
    file_path: str = Field(..., description="Absolute path: /data/patches/{id}.png")
    bbox: dict = Field(sa_column=Column(JSON), description='{"x","y","width","height"}')