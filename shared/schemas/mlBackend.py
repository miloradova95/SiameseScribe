# shared/schemas/ml_backend.py
# Pydantic schemas for the ML Backend API.
# These define what goes in and out of every ML backend endpoint.

from pydantic import BaseModel, Field
from typing import Annotated, Literal
from shared.schemas.shared import BoundingBox


# 128-dim embedding vector — matches SiameseNetwork output (no autoencoder reduction, as discussed with djordje)
Embedding128 = Annotated[list[float], Field(min_length=128, max_length=128)]


# ─────────────────────────────────────────────
# POST /segment
# ─────────────────────────────────────────────

class SegmentRequest(BaseModel):
    image_path: str


class SegmentedPatch(BaseModel):
    patch_id: int
    bbox: BoundingBox
    patch_path: str


class SegmentResponse(BaseModel):
    patches: list[SegmentedPatch]


# ─────────────────────────────────────────────
# POST /embed_patches
# ─────────────────────────────────────────────

class EmbedPatchesRequest(BaseModel):
    patch_paths: list[str] = Field(..., min_length=1)


class PatchEmbedding(BaseModel):
    patch_path: str
    vector: Embedding128


class EmbedPatchesResponse(BaseModel):
    embeddings: list[PatchEmbedding]


# ─────────────────────────────────────────────
# POST /search_patches
# ─────────────────────────────────────────────

class SearchPatchesRequest(BaseModel):
    embedding: Embedding128
    top_k: int = Field(4, ge=1, le=20)


class SearchResultItem(BaseModel):
    patch_id: int
    similarity_score: float = Field(..., ge=0.0, le=1.0)


class SearchPatchesResponse(BaseModel):
    results: list[SearchResultItem]


# ─────────────────────────────────────────────
# POST /explain_pair
# ─────────────────────────────────────────────

class ExplainPairRequest(BaseModel):
    query_patch_path: str
    result_patch_path: str


class ExplainPairHeatmaps(BaseModel):
    query: str = Field(..., description="Path to heatmap for query patch")
    result: str = Field(..., description="Path to heatmap for result patch")


class ExplainPairResponse(BaseModel):
    heatmaps: ExplainPairHeatmaps


# ─────────────────────────────────────────────
# POST /embed_all_patches
# ─────────────────────────────────────────────

class EmbedAllPatchesResponse(BaseModel):
    status: Literal["started"]
    message: str


# ─────────────────────────────────────────────
# POST /retrain
# ─────────────────────────────────────────────

class Triplet(BaseModel):
    anchor_patch_path: str
    positive_patch_path: str
    negative_patch_path: str


class RetrainRequest(BaseModel):
    triplets: list[Triplet] = Field(..., min_length=1)


class RetrainResponse(BaseModel):
    status: Literal["training_started"]