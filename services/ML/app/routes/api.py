import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[3]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from PIL import Image
from torchvision import transforms

from services.ML.app.services.Finetune import count_constructable_triplets, finetune
from services.ML.app.services.segment import extract_patches
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
    SegmentedPatch,
)
from shared.schemas.shared import BoundingBox

router = APIRouter()

_embed_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# 1. SEGMENT IMAGE
# ─────────────────────────────────────────────

@router.post("/segment", response_model=SegmentResponse)
def segment_image(req: SegmentRequest, request: Request):
    seg_service = request.app.state.seg_service
    if seg_service is None:
        raise HTTPException(status_code=503, detail="Segmentation model not loaded")

    image_path = Path(req.image_path)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path

    image_bytes = image_path.read_bytes()
    mask_array, _ = seg_service.predict_mask(image_bytes)

    # blob_removal returns H×W×3 (all channels identical) — reduce to single channel
    if mask_array.ndim == 3:
        mask_array = mask_array[:, :, 0]
    mask_pil = Image.fromarray(mask_array, mode="L")

    # Save mask PNG alongside dataset masks for reproducibility
    mask_dir = PROJECT_ROOT / "data" / "dataset" / "masks" / "uploads"
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_pil.save(mask_dir / f"{image_path.stem}.png")

    output_dir = PROJECT_ROOT / "data" / "patches" / "uploads"
    patches_data = extract_patches(
        image_path=str(image_path),
        patch_size=(128, 128),
        step_size=64,
        output_dir=str(output_dir),
        mask_image=mask_pil,
        threshold=0.1,
    )

    patches = [
        SegmentedPatch(
            patch_index=p["patch_id"],
            bbox=BoundingBox(x=p["bbox"][0], y=p["bbox"][1], width=p["bbox"][2], height=p["bbox"][3]),
            patch_path=str(Path(p["patch_path"]).relative_to(PROJECT_ROOT)).replace("\\", "/"),
        )
        for p in patches_data
    ]
    return {"patches": patches}


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
def search_patches(req: SearchPatchesRequest, request: Request):
    collection = request.app.state.collection
    if collection is None:
        raise HTTPException(status_code=503, detail="ChromaDB collection not loaded — run Embedd.py first")

    results = collection.query(
        query_embeddings=[req.embedding],
        n_results=req.top_k,
        include=["distances"],
    )

    ids = results["ids"][0]
    distances = results["distances"][0]

    # ChromaDB cosine space: distance = 1 − cosine_similarity
    return {
        "results": [
            {"patch_filename": pid, "similarity_score": round(1.0 - dist, 6)}
            for pid, dist in zip(ids, distances)
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
def retrain(req: RetrainRequest, background_tasks: BackgroundTasks):
    feedback_dicts = [f.model_dump() for f in req.feedback]
    triplets_used = count_constructable_triplets(feedback_dicts, req.k_triplets)

    if triplets_used > 0:
        background_tasks.add_task(finetune, feedback_dicts, req.k_triplets)

    return {"status": "training_started", "triplets_used": triplets_used}
