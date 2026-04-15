import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[3]
sys.path.append(str(PROJECT_ROOT))

import torch
from fastapi import APIRouter, Request
from PIL import Image
from torchvision import transforms

from shared.schemas.mlBackend import (
    EmbedAllPatchesResponse,
    EmbedPatchesRequest,
    EmbedPatchesResponse,
    ExplainPairRequest,
    ExplainPairResponse,
    RetrainRequest,
    RetrainResponse,
    SearchPatchesRequest,
    SearchPatchesResponse,
    SegmentRequest,
    SegmentResponse,
)

router = APIRouter()

_embed_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# 1. SEGMENT IMAGE
# ─────────────────────────────────────────────

@router.post("/segment", response_model=SegmentResponse)
def segment_image(req: SegmentRequest):
    return {
        "patches": [
            {"patch_id": 5001, "bbox": {"x": 10, "y": 20, "width": 100, "height": 100}, "patch_path": "/data/patches/5001.png"},
            {"patch_id": 5002, "bbox": {"x": 120, "y": 50, "width": 100, "height": 100}, "patch_path": "/data/patches/5002.png"},
        ]
    }


# ─────────────────────────────────────────────
# 2. EMBED PATCHES
# ─────────────────────────────────────────────

@router.post("/embed_patches", response_model=EmbedPatchesResponse)
def embed_patches(req: EmbedPatchesRequest, request: Request):
    model = request.app.state.model
    device = request.app.state.device

    embeddings = []
    for patch_path in req.patch_paths:
        resolved = Path(patch_path)
        if not resolved.is_absolute():
            resolved = PROJECT_ROOT / resolved
        image = Image.open(resolved).convert("RGB")
        tensor = _embed_transforms(image).unsqueeze(0).to(device)

        with torch.no_grad():
            vector = model.get_embedding(tensor).cpu().squeeze(0).tolist()

        embeddings.append({"patch_path": str(patch_path), "vector": vector})

    return {"embeddings": embeddings}


# ─────────────────────────────────────────────
# 3. EMBED ALL PATCHES (initial batch embedding)
# ─────────────────────────────────────────────

@router.post("/embed_all_patches", response_model=EmbedAllPatchesResponse)
def embed_all_patches():
    """
    Triggers batch embedding of all pre-extracted patches into ChromaDB.

    This endpoint is intentionally NOT routed through the main backend for the initial
    population — calling /embed_patches per-patch would mean ~90,000 HTTP round trips.
    Instead, run the standalone script directly:

        python services/ML/app/services/Embedd.py --collection <name> [--model <path>]

    This endpoint exists so the process can optionally be triggered remotely once
    the batch script is wired up as a background task.
    """
    return {
        "status": "started",
        "message": (
            "Not yet implemented as a live endpoint. "
            "Run Embedd.py directly: "
            "python services/ML/app/services/Embedd.py --collection <name>"
        ),
    }


# ─────────────────────────────────────────────
# 4. SEARCH SIMILAR PATCHES
# ─────────────────────────────────────────────

@router.post("/search_patches", response_model=SearchPatchesResponse)
def search_patches(req: SearchPatchesRequest):
    return {
        "results": [
            {"patch_id": 6001, "similarity_score": 0.87},
            {"patch_id": 2134, "similarity_score": 0.82},
        ]
    }


# ─────────────────────────────────────────────
# 5. PAIRWISE HEATMAP
# ─────────────────────────────────────────────

@router.post("/explain_pair", response_model=ExplainPairResponse)
def explain_pair(req: ExplainPairRequest):
    return {
        "heatmaps": {
            "query": "/data/heatmaps/q5001_r6001.png",
            "result": "/data/heatmaps/r6001_q5001.png",
        }
    }


# ─────────────────────────────────────────────
# 6. RETRAIN MODEL
# ─────────────────────────────────────────────

@router.post("/retrain", response_model=RetrainResponse)
def retrain(req: RetrainRequest):
    return {"status": "training_started"}
