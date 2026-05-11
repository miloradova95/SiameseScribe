from datetime import datetime
from typing import Optional
from pydantic import BaseModel, field_validator

from services.backend.schemas._datetime import ensure_utc


class FinetuneRunItem(BaseModel):
    id: int
    triggered_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    status: str
    trigger_source: str
    t_real: int
    t_aug: int
    p_pos: int
    triplets_used: int
    mlflow_run_id: Optional[str]
    feedback_count: int
    error_msg: Optional[str]

    @field_validator("triggered_at", "started_at", "completed_at", mode="before")
    @classmethod
    def _normalize_dt(cls, v):
        if v is None:
            return v
        return ensure_utc(v)


class FinetuneRunTriggerResponse(BaseModel):
    status: str          # "triggered" | "skipped"
    reason: Optional[str] = None
    run_id: Optional[int] = None
