from datetime import datetime, timezone
from typing import Optional
from sqlmodel import SQLModel, Field


class FinetuneRun(SQLModel, table=True):
    __tablename__ = "finetune_runs"

    id: Optional[int] = Field(default=None, primary_key=True)
    triggered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    status: str = Field(default="pending")      # pending | running | completed | failed
    trigger_source: str = Field(default="auto") # auto | manual
    t_real: int = Field(default=0)
    t_aug: int = Field(default=0)
    p_pos: int = Field(default=0)
    triplets_used: int = Field(default=0)
    mlflow_run_id: Optional[str] = Field(default=None)
    feedback_ids: str = Field(default="[]")     # JSON-encoded list[int]
    error_msg: Optional[str] = Field(default=None)
