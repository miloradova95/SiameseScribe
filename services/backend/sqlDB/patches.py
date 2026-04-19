from sqlmodel import SQLModel, Field, Column
from sqlalchemy import JSON
from typing import Optional


class Patch(SQLModel, table=True):
    __tablename__ = "patches"

    id: Optional[int] = Field(default=None, primary_key=True)
    source_image_id: int = Field(..., foreign_key="images.id", index=True)
    file_path: str = Field(..., description="Absolute path: /data/patches/{id}.png")
    bbox: dict = Field(sa_column=Column(JSON), description='{"x","y","width","height"}')
    group: Optional[str] = Field(default=None, description="Dataset group (e.g. A, B)")
    codex: Optional[str] = Field(default=None, description="Codex identifier (e.g. ccl71)")
    pen_flourishing_percent: Optional[float] = Field(default=None, description="Fraction of patch covered by pen flourishing")