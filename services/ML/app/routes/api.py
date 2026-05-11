import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[3]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch
from fastapi import APIRouter, HTTPException, Request
from PIL import Image
from torchvision import transforms

from services.ML.app.services.Finetune import count_samples, finetune
from services.ML.app.services.HeatmapUtil import save_heatmap_overlay
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


def _resolve_path(path_str: str) -> Path:
    
    """
    Resolve a patch path to an absolute filesystem path.

    Paths stored in the DB are relative to PROJECT_ROOT.parent
    (e.g. "SiameseScribe/data/patches/test/foo.png").
    Paths coming from the segment endpoint are relative to PROJECT_ROOT.
    Handle both cases by trying PROJECT_ROOT.parent first, then PROJECT_ROOT.
    """
    
    normalized = path_str.replace("\\", "/")

    p = Path(normalized)
    if p.is_absolute():
        return p

    candidate = PROJECT_ROOT.parent / p
    if candidate.exists():
        return candidate

    candidate2 = PROJECT_ROOT / p
    if candidate2.exists():
        return candidate2

    # Return the most-likely path so the caller gets a clear FileNotFoundError
    return PROJECT_ROOT.parent / p


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

    if mask_array.ndim == 3:
        mask_array = mask_array[:, :, 0]
    mask_pil = Image.fromarray(mask_array, mode="L")

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
        resolved = _resolve_path(patch_path)
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

    where = {"source_image_id": {"$ne": req.exclude_source_image_id}} \
        if req.exclude_source_image_id is not None else None

    results = collection.query(
        query_embeddings=[req.embedding],
        n_results=req.top_k,
        include=["distances"],
        **({"where": where} if where else {}),
    )

    ids = results["ids"][0]
    distances = results["distances"][0]

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
def explain_pair(req: ExplainPairRequest, request: Request):
    model = request.app.state.model
    device = request.app.state.device

    query_path = _resolve_path(req.query_patch_path)
    if not query_path.exists():
        raise HTTPException(status_code=404, detail=f"Query patch not found: {query_path}")
    result_path = _resolve_path(req.result_patch_path)
    if not result_path.exists():
        raise HTTPException(status_code=404, detail=f"Result patch not found: {result_path}")

    query_id = Path(req.query_patch_path).stem
    result_id = Path(req.result_patch_path).stem

    query_pil = Image.open(query_path).convert("RGB")
    result_pil = Image.open(result_path).convert("RGB")

    query_w, query_h = query_pil.size

    query_tensor = _embed_transforms(query_pil).unsqueeze(0).to(device)
    result_tensor = _embed_transforms(result_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, sfam = model.forward_with_sfam(
            query_tensor, result_tensor, output_size=(query_h, query_w)
        )

    heatmaps_dir = PROJECT_ROOT / "data" / "heatmaps"
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    query_output_path = heatmaps_dir / f"q{query_id}_r{result_id}.png"
    result_output_path = heatmaps_dir / f"r{result_id}_q{query_id}.png"

    save_heatmap_overlay(query_pil, sfam, query_output_path)
    save_heatmap_overlay(result_pil, sfam, result_output_path)

    return {
        "heatmaps": {
            "query": query_output_path.relative_to(PROJECT_ROOT).as_posix(),
            "result": result_output_path.relative_to(PROJECT_ROOT).as_posix(),
        }
    }


# ─────────────────────────────────────────────
# 6. RETRAIN MODEL
# ─────────────────────────────────────────────

@router.post("/retrain", response_model=RetrainResponse)
def retrain(req: RetrainRequest):
    feedback_dicts = [f.model_dump() for f in req.feedback]
    samples = count_samples(feedback_dicts)

    if not samples["can_finetune"]:
        return {
            "status": "skipped",
            "triplets_used": 0,
            "run_id": None,
            "t_real": samples["t_real"],
            "t_aug": samples["t_aug"],
            "p_pos": samples["p_pos"],
        }

    result = finetune(feedback_dicts, req.k_triplets)
    return {
        "status": "completed",
        "triplets_used": result["triplets_used"],
        "run_id": result["run_id"] or None,
        "t_real": result["t_real"],
        "t_aug": result["t_aug"],
        "p_pos": result["p_pos"],
    }
