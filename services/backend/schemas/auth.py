from pydantic import BaseModel, field_validator
from datetime import datetime

from services.backend.schemas._datetime import ensure_utc


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    role: str = "user"


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
    is_active: bool
    created_at: datetime

    @field_validator("created_at", mode="before")
    @classmethod
    def _normalize_created_at(cls, value: datetime) -> datetime:
        return ensure_utc(value)
