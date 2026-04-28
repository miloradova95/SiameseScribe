from typing import Optional
from datetime import datetime, timezone
from sqlmodel import SQLModel, Field

class Feedback(SQLModel, table=True):
    __tablename__ = "feedback"

    id: Optional[int] = Field(default=None, primary_key=True)
    query_patch_id: int = Field(..., foreign_key="patches.id", index=True)
    result_patch_id: int = Field(..., foreign_key="patches.id", index=True)
    label: int = Field(..., description="1 = similar | 0 = not_similar")
    used_for_retrain: bool = Field(default=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))