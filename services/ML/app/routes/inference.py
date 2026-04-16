from fastapi import APIRouter

# Shared schemas (adjust path depending on your project structure)
from shared.schemas.ml import (
    SegmentRequest,
    SegmentResponse,
    EmbedRequest,
    EmbedResponse,
    SearchRequest,
    SearchResponse,
    ExplainRequest,
    HeatmapResponse,
    TrainRequest,
    TrainResponse,
)

router = APIRouter()


# =========================
# 1. SEGMENT IMAGE
# =========================
@router.post("/segment", response_model=SegmentResponse)
def segment_image(req: SegmentRequest):
    """
    Splits image into patches using segmentation model
    """

    # from app.services import segment
    # return segment.run(req.image_path)

    return {
        "patches": [
            {
                "patch_id": 5001,
                "bbox": [10, 20, 100, 100],
                "patch_path": "/data/patches/5001.png"
            },
            {
                "patch_id": 5002,
                "bbox": [120, 50, 100, 100],
                "patch_path": "/data/patches/5002.png"
            }
        ]
    }

    # raise NotImplementedError("Segmentation not implemented yet")


# =========================
# 2. EMBED PATCHES
# =========================
@router.post("/embed_patches", response_model=EmbedResponse)
def embed_patches(req: EmbedRequest):
    """
    Generate embeddings for patches
    """

    # from app.services import embed
    # return embed.run(req.patch_paths)

    return {
        "embeddings": [
            {
                "patch_path": req.patch_paths[0] if req.patch_paths else "",
                "vector": [0.1] * 32
            }
        ]
    }

    # raise NotImplementedError("Embedding not implemented yet")


# =========================
# 3. SEARCH SIMILAR PATCHES
# =========================
@router.post("/search_patches", response_model=SearchResponse)
def search_patches(req: SearchRequest):
    """
    Find nearest neighbor patches
    """

    # from app.services import search
    # return search.run(req.embedding, req.top_k)

    return {
        "results": [
            {"patch_id": 6001, "similarity_score": 0.87},
            {"patch_id": 2134, "similarity_score": 0.82}
        ]
    }

    # raise NotImplementedError("Search not implemented yet")


# =========================
# 4. PAIRWISE HEATMAP
# =========================
@router.post("/explain_pair", response_model=HeatmapResponse)
def explain_pair(req: ExplainRequest):
    """
    Generate explanation heatmaps for a patch pair
    """

    # from app.services import explain
    # return explain.run(req.query_patch_path, req.result_patch_path)

    return {
        "heatmaps": {
            "query": "/data/heatmaps/q5001_r6001.png",
            "result": "/data/heatmaps/r6001_q5001.png"
        }
    }

    # raise NotImplementedError("Explain not implemented yet")


# =========================
# 5. RETRAIN MODEL
# =========================
@router.post("/retrain", response_model=TrainResponse)
def retrain(req: TrainRequest):
    """
    Retrain / fine-tune Siamese network
    """

    # from app.services import train
    # return train.run(req.triplets)

    return {
        "status": "training_started"
    }

    # raise NotImplementedError("Training not implemented yet")