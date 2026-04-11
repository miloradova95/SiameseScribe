# shared/schemas/main_backend.py
# Pydantic schemas for the Main Backend API.
# These define what goes in and out of every main backend endpoint.

from pydantic import BaseModel, Field
from typing import Literal
from enum import Enum
from shared.schemas.shared import BoundingBox


# ─────────────────────────────────────────────
# POST /images
# ─────────────────────────────────────────────

class PatchSummary(BaseModel):
    patch_id: int
    bbox: BoundingBox


class UploadImageResponse(BaseModel):
    image_id: int
    patches: list[PatchSummary]


# ─────────────────────────────────────────────
# POST /patches/{patch_id}/search
# ─────────────────────────────────────────────

class SearchRequest(BaseModel):
    top_k: int = Field(4, ge=1, le=20)


class HeatmapPair(BaseModel):
    query: str = Field(..., description="URL: GET /heatmaps/{q}/{r}/query")
    result: str = Field(..., description="URL: GET /heatmaps/{q}/{r}/result")


class QueryPatchInfo(BaseModel):
    patch_id: int
    image_id: int
    image_url: str
    bbox: BoundingBox


class SearchResultItem(BaseModel):
    patch_id: int
    image_id: int
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    image_url: str
    patch_bbox: BoundingBox
    heatmaps: HeatmapPair


class SearchResponse(BaseModel):
    query_patch: QueryPatchInfo
    results: list[SearchResultItem]


# ─────────────────────────────────────────────
# POST /feedback
# ─────────────────────────────────────────────

class FeedbackLabel(str, Enum):
    similar = "similar"
    not_similar = "not_similar"


class FeedbackRequest(BaseModel):
    query_patch_id: int
    result_patch_id: int
    label: FeedbackLabel


class FeedbackResponse(BaseModel):
    status: Literal["ok"]


# ─────────────────────────────────────────────
# POST /retrain
# ─────────────────────────────────────────────

class RetrainResponse(BaseModel):
    status: Literal["training_started"]


# ─────────────────────────────────────────────
# GET /heatmaps/{query_patch_id}/{result_patch_id}/{type}
# ─────────────────────────────────────────────

class HeatmapType(str, Enum):
    query = "query"
    result = "result"