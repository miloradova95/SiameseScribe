from enum import Enum
from datetime import datetime

from pydantic import BaseModel


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
