import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[3]
sys.path.append(str(PROJECT_ROOT))

import mlflow
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from services.ML.app.services.SiameseNetwork import SiameseNetwork
from services.ML.app.chroma_client import get_chroma_client
from services.ML.app.services.mlflow_utils import fix_mlflow_paths

METADATA_TEST  = PROJECT_ROOT / "data" / "patches" / "patches_test_metadata.csv"
METADATA_TRAIN = PROJECT_ROOT / "data" / "patches" / "patches_train_metadata.csv"
PATCHES_DIR    = PROJECT_ROOT / "data" / "patches" / "train"
PATCHES_TEST_DIR = PROJECT_ROOT / "data" / "patches" / "test"
MODEL_PATH     = PROJECT_ROOT / "data" / "models" / "trainedModel.pth"
CHROMA_PATH    = str(PROJECT_ROOT / "data" / "chromaDB" / "data" / "chroma_store")
MLFLOW_DIR     = PROJECT_ROOT / "data" / "mlruns"

EMBEDDING_DIM = 128
EMBED_BATCH   = 512
QUERY_CHUNK   = 256

_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def precision_at_k(ranked_labels: list[str], true_label: str, k: int) -> float:
    return sum(1 for l in ranked_labels[:k] if l == true_label) / k


def average_precision(ranked_labels: list[str], true_label: str, k: int) -> float:
    hits, ap = 0, 0.0
    for i, label in enumerate(ranked_labels[:k]):
        if label == true_label:
            hits += 1
            ap += hits / (i + 1)
    return ap / hits if hits else 0.0


# ─────────────────────────────────────────────
# Shared: batched embedding
# ─────────────────────────────────────────────

def _embed_batched(model, patch_paths: list[Path], device: str) -> torch.Tensor:
    """Embed a list of patch paths in batches. Returns (N, 128) CPU tensor."""
    model.eval()
    all_embs = []
    for i in tqdm(range(0, len(patch_paths), EMBED_BATCH), desc="Embedding", unit="batch"):
        batch_paths = patch_paths[i:i + EMBED_BATCH]
        imgs = [_transforms(Image.open(p).convert("RGB")) for p in batch_paths]
        tensor = torch.stack(imgs).to(device)
        with torch.no_grad():
            emb = model.get_embedding(tensor).cpu()
        all_embs.append(emb)
    return torch.cat(all_embs)  # (N, 128)


# ─────────────────────────────────────────────
# Mode 1: Full eval — ChromaDB-backed
# ─────────────────────────────────────────────

def evaluate_full(collection, model, device: str, top_k: int = 5, mlflow_run_id: str = None, exclude_same_image: bool = True) -> dict:
    """
    Query every test patch against ChromaDB and compute P@K + mAP.
    Requires Embedd.py to have been run with the current model weights.
    Test embeddings are computed in-memory and discarded after evaluation.
    """
    df = pd.read_csv(METADATA_TEST)

    # Resolve patch paths — try test dir first, fall back to train dir
    patch_paths = []
    for fname in df["patch_filename"]:
        p = PATCHES_TEST_DIR / fname
        if not p.exists():
            p = PATCHES_DIR / fname
        patch_paths.append(p)

    query_embs = _embed_batched(model, patch_paths, device)  # (N, 128)
    groups = df["group"].tolist()
    source_image_ids = df["source_image_id"].tolist()

    precisions, aps, per_query = [], [], []

    print(f"Querying ChromaDB (top_k={top_k}, exclude_same_image={exclude_same_image}) for {len(patch_paths)} test patches...")
    for i, (emb, true_group, fname, src_img_id) in enumerate(
        tqdm(zip(query_embs, groups, df["patch_filename"], source_image_ids), total=len(patch_paths), desc="Evaluating")
    ):
        where = {"source_image_id": {"$ne": int(src_img_id)}} if exclude_same_image else None

        results = collection.query(
            query_embeddings=[emb.tolist()],
            n_results=top_k,
            include=["metadatas"],
            **({"where": where} if where else {}),
        )
        retrieved_groups = [m["group"] for m in results["metadatas"][0]]

        p = precision_at_k(retrieved_groups, true_group, top_k)
        ap = average_precision(retrieved_groups, true_group, top_k)
        precisions.append(p)
        aps.append(ap)
        per_query.append({"patch": fname, "group": true_group, "precision_at_k": p, "ap": ap})

    mean_p = sum(precisions) / len(precisions)
    mean_ap = sum(aps) / len(aps)

    summary = {
        "precision_at_k": round(mean_p, 6),
        "mAP": round(mean_ap, 6),
        "top_k": top_k,
        "exclude_same_image": exclude_same_image,
        "num_samples": len(patch_paths),
        "per_query": per_query,
    }

    # Save JSON artifact
    results_dir = PROJECT_ROOT / "data" / "models" / "eval_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    label = mlflow_run_id[:8] if mlflow_run_id else "standalone"
    out_path = results_dir / f"{label}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {out_path}")

    # Log to MLflow if a run is active or an ID was provided
    _log_to_mlflow(
        {"eval/precision_at_k": mean_p, "eval/mAP": mean_ap},
        artifact_path=str(out_path),
        parent_run_id=mlflow_run_id,
    )

    return {k: v for k, v in summary.items() if k != "per_query"}


# ─────────────────────────────────────────────
# Mode 2: Quick eval — in-memory, per-epoch
# ─────────────────────────────────────────────

def evaluate_quick(
    model,
    device: str,
    top_k: int = 5,
    n_gallery_per_group: int = 200,
    n_queries: int = 1000,
    epoch: int = None,
) -> dict:
    """
    In-memory evaluation using sampled train gallery + sampled test queries.
    Does NOT use ChromaDB — gallery embeddings are recomputed with current model weights,
    so metrics reflect the model at the current epoch rather than a stale checkpoint.
    Logs to the active MLflow run if one exists.
    """
    train_df = pd.read_csv(METADATA_TRAIN)
    test_df  = pd.read_csv(METADATA_TEST)

    # Sample gallery: n_gallery_per_group patches per group from train set
    gallery_df = (
        train_df.groupby("group", group_keys=False)
        .apply(lambda g: g.sample(min(n_gallery_per_group, len(g)), random_state=42))
        .reset_index(drop=True)
    )

    # Sample queries: n_queries test patches, stratified by group
    n_groups = test_df["group"].nunique()
    per_group = max(1, n_queries // n_groups)
    query_df = (
        test_df.groupby("group", group_keys=False)
        .apply(lambda g: g.sample(min(per_group, len(g)), random_state=42))
        .reset_index(drop=True)
    )

    gallery_paths = [PATCHES_DIR / f for f in gallery_df["patch_filename"]]
    query_paths   = []
    for fname in query_df["patch_filename"]:
        p = PATCHES_TEST_DIR / fname
        if not p.exists():
            p = PATCHES_DIR / fname
        query_paths.append(p)

    gallery_embs = _embed_batched(model, gallery_paths, device)   # (G, 128)
    query_embs   = _embed_batched(model, query_paths, device)     # (Q, 128)

    # L2-normalise for cosine similarity via dot product
    gallery_norm = F.normalize(gallery_embs, p=2, dim=1)  # (G, 128)
    query_norm   = F.normalize(query_embs,   p=2, dim=1)  # (Q, 128)

    gallery_groups = gallery_df["group"].tolist()
    query_groups   = query_df["group"].tolist()

    precisions, aps = [], []

    for chunk_start in range(0, len(query_norm), QUERY_CHUNK):
        chunk = query_norm[chunk_start:chunk_start + QUERY_CHUNK]  # (C, 128)
        sims  = chunk @ gallery_norm.T                             # (C, G)

        for i, row in enumerate(sims):
            true_group = query_groups[chunk_start + i]
            ranked_idx = row.argsort(descending=True).tolist()
            ranked_labels = [gallery_groups[j] for j in ranked_idx[:top_k]]

            precisions.append(precision_at_k(ranked_labels, true_group, top_k))
            aps.append(average_precision(ranked_labels, true_group, top_k))

    mean_p  = sum(precisions) / len(precisions)
    mean_ap = sum(aps) / len(aps)

    metrics = {"precision_at_k": round(mean_p, 6), "mAP": round(mean_ap, 6)}

    # Log to active MLflow run if one exists
    if mlflow.active_run():
        step = epoch if epoch is not None else None
        mlflow.log_metrics(
            {"eval/precision_at_k": mean_p, "eval/mAP": mean_ap},
            step=step,
        )

    return metrics


# ─────────────────────────────────────────────
# MLflow helper
# ─────────────────────────────────────────────

def _log_to_mlflow(metrics: dict, artifact_path: str = None, parent_run_id: str = None):
    if mlflow.active_run():
        mlflow.log_metrics(metrics)
        if artifact_path:
            mlflow.log_artifact(artifact_path, artifact_path="eval")
    elif parent_run_id:
        # Reopen the existing training run and log directly to it — eval metrics
        # appear on the same run row in the MLflow UI alongside train_loss.
        fix_mlflow_paths(MLFLOW_DIR, "siamese-scribe")
        mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
        with mlflow.start_run(run_id=parent_run_id):
            mlflow.log_metrics(metrics)
            if artifact_path:
                mlflow.log_artifact(artifact_path, artifact_path="eval")


# ─────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate model P@K and mAP against ChromaDB")
    parser.add_argument("--collection", type=str, required=True, help="ChromaDB collection name")
    parser.add_argument("--model",      type=str, default=str(MODEL_PATH), help="Path to model checkpoint")
    parser.add_argument("--top_k",      type=int, default=5)
    parser.add_argument("--mlflow_run_id", type=str, default=None,
                        help="Parent MLflow run ID — eval metrics will be logged as a child run")
    parser.add_argument("--no_exclude_same_image", action="store_true",
                        help="Include patches from the same source image in results (off by default)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = SiameseNetwork(embedding_dim=EMBEDDING_DIM).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"Model loaded from {args.model}")

    client = get_chroma_client(CHROMA_PATH)
    collection = client.get_collection(args.collection)
    print(f"Collection '{args.collection}' loaded ({collection.count()} embeddings)")

    metrics = evaluate_full(
        collection, model, device,
        top_k=args.top_k,
        mlflow_run_id=args.mlflow_run_id,
        exclude_same_image=not args.no_exclude_same_image,
    )

    print(f"\n=== Evaluation Results ===")
    print(f"Samples:       {metrics['num_samples']}")
    print(f"Precision@{args.top_k}:  {metrics['precision_at_k']:.4f}")
    print(f"mAP:           {metrics['mAP']:.4f}")


if __name__ == "__main__":
    main()
