from fastapi import APIRouter

from shared.schemas.mlBackend import (
    SegmentRequest,
    SegmentResponse,
    EmbedPatchesRequest,
    EmbedPatchesResponse,
    SearchPatchesRequest,
    SearchPatchesResponse,
    ExplainPairRequest,
    ExplainPairResponse,
    RetrainRequest,
    RetrainResponse,
)


router = APIRouter()


# =========================
# 1. SEGMENT IMAGE
# =========================
@router.post("/segment", response_model=SegmentResponse)
def segment_image(req: SegmentRequest):

    return {
        "patches": [
            {
                "patch_id": 5001,
                "bbox": {
                    "x": 10,
                    "y": 20,
                    "width": 100,
                    "height": 100
                },
                "patch_path": "/data/patches/5001.png"
            },
            {
                "patch_id": 5002,
                "bbox": {
                    "x": 120,
                    "y": 50,
                    "width": 100,
                    "height": 100
                },
                "patch_path": "/data/patches/5002.png"
            }
        ]
    }


# =========================
# 2. EMBED PATCHES
# =========================
@router.post("/embed_patches", response_model=EmbedPatchesResponse)
def embed_patches(req: EmbedPatchesRequest):

    return {
        "embeddings": [
            {
                "patch_path": req.patch_paths[0],
                "vector": [0.1] * 32  # MUST be 32 values
            }
        ]
    }


# =========================
# 3. SEARCH SIMILAR PATCHES
# =========================
@router.post("/search_patches", response_model=SearchPatchesResponse)
def search_patches(req: SearchPatchesRequest):

    return {
        "results": [
            {"patch_id": 6001, "similarity_score": 0.87},
            {"patch_id": 2134, "similarity_score": 0.82}
        ]
    }


# =========================
# 4. PAIRWISE HEATMAP
# =========================
@router.post("/explain_pair", response_model=ExplainPairResponse)
def explain_pair(req: ExplainPairRequest):

    return {
        "heatmaps": {
            "query": "/data/heatmaps/q5001_r6001.png",
            "result": "/data/heatmaps/r6001_q5001.png"
        }
    }


# =========================
# 5. RETRAIN MODEL
# =========================
@router.post("/retrain", response_model=RetrainResponse)
def retrain(req: RetrainRequest):

    return {
        "status": "training_started"
    }