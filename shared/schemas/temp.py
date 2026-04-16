from typing import Optional
from datetime import datetime, timezone
from pydantic import BaseModel, EmailStr
from sqlmodel import SQLModel, Field

# ── User table ────────────────────────────────
class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(unique=True, index=True)
    email: str = Field(unique=True, index=True)
    hashed_password: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Feedback(SQLModel, table=True):
    __tablename__ = "feedback"

    id: Optional[int] = Field(default=None, primary_key=True)
    query_patch_id: int = Field(..., foreign_key="patches.id", index=True)
    result_patch_id: int = Field(..., foreign_key="patches.id", index=True)
    label: int = Field(..., description="1 = similar | 0 = not_similar")
    used_for_retrain: bool = Field(default=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class DataPatch(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    file_path: str = Field(index=True)
    source_image: str = Field(index=True)
    group: str = Field(index=True)

# ── Schemas ───────────────────────────────────
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime


class DataPatchCreate(BaseModel):
    file_path: str
    source_image: str
    group: str


class DataPatchResponse(BaseModel):
    id: int
    file_path: str
    source_image: str
    group: str