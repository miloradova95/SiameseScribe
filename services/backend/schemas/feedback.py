from enum import Enum
from datetime import datetime

from pydantic import BaseModel, Field


class FeedbackLabel(str, Enum):
    similar = "similar"
    not_similar = "not_similar"


class FeedbackCreate(BaseModel):
    query_patch_id: int
    result_patch_id: int
    label: FeedbackLabel


class FeedbackCreateResponse(BaseModel):
    id: int
    status: str = "ok"


class FeedbackListItem(BaseModel):
    id: int
    query_patch_id: int
    result_patch_id: int
    query_patch_file_name: str
    result_patch_file_name: str
    label: FeedbackLabel
    created_at: datetime
    used_for_retrain: bool


class AdminFeedbackListItem(FeedbackListItem):
    user_id: int
    username: str
    user_email: str


class AdminFeedbackFilterParams(BaseModel):
    user_id: int | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    used_for_retrain: bool | None = None


class FeedbackRetrainRequest(BaseModel):
    feedback_ids: list[int] = Field(..., min_length=1)
    k_triplets: int = Field(default=1, ge=1, le=5)


class FeedbackRetrainResponse(BaseModel):
    status: str = "training_started"
    feedback_count: int
