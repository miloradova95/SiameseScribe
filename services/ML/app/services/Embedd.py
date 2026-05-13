import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[3]
sys.path.append(str(PROJECT_ROOT))

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from services.ML.app.services.SiameseNetwork import SiameseNetwork
from services.ML.app.chroma_client import get_chroma_client, get_or_create_collection

PATCHES_DIR  = PROJECT_ROOT / "data" / "patches" / "train"
METADATA_CSV = PROJECT_ROOT / "data" / "patches" / "patches_train_metadata.csv"
CHROMA_PATH  = str(PROJECT_ROOT / "data" / "chromaDB" / "data" / "chroma_store")
MODEL_PATH   = PROJECT_ROOT / "data" / "models" / "trainedModel.pth"

BATCH_SIZE    = 512
EMBEDDING_DIM = 128

_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(model_path: Path, device: str) -> SiameseNetwork:
    model = SiameseNetwork(embedding_dim=EMBEDDING_DIM).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


def embed_patches(model, patch_paths: list[Path], device: str) -> list[list[float]]:
    embeddings = []
    for patch_path in tqdm(patch_paths, desc="Embedding patches"):
        image = Image.open(patch_path).convert("RGB")
        tensor = _transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.get_embedding(tensor).cpu().squeeze(0).tolist()
        embeddings.append(emb)
    return embeddings


def store_embeddings(
    collection,
    patch_paths: list[Path],
    embeddings: list[list[float]],
    metadatas: list[dict],
):
    """
    Upsert embeddings into ChromaDB in batches.

    Uses patch filename as the deterministic ChromaDB ID, so re-running this
    script with the same collection name is safe — existing entries are updated
    in place rather than duplicated.
    """
    ids = [p.name for p in patch_paths]

    for i in range(0, len(ids), BATCH_SIZE):
        collection.upsert(
            ids=ids[i:i + BATCH_SIZE],
            embeddings=embeddings[i:i + BATCH_SIZE],
            metadatas=metadatas[i:i + BATCH_SIZE],
        )
        print(f"  Stored {min(i + BATCH_SIZE, len(ids))}/{len(ids)}")

    print(f"Done — {len(ids)} embeddings in collection '{collection.name}'")


UPLOADS_DIR = PROJECT_ROOT / "data" / "patches" / "uploads"


def embed_and_upsert(model, device: str, collection) -> int:
    """
    Re-embed all patches (training + uploaded) and upsert into ChromaDB.
    Safe to call repeatedly — uses upsert, so existing entries are updated in place.
    Returns the total number of patches embedded.
    """
    import pandas as pd

    total = 0

    # ── Training patches from metadata CSV ──────────────────────────────────
    if METADATA_CSV.exists():
        df = pd.read_csv(METADATA_CSV)
        train_paths = [PATCHES_DIR / row["patch_filename"] for _, row in df.iterrows()]
        train_meta = [
            {
                "source_image":            row["source_image"],
                "source_image_id":         int(row["source_image_id"]),
                "group":                   row["group"],
                "codex":                   row["codex"],
                "x":                       int(row["x"]),
                "y":                       int(row["y"]),
                "pen_flourishing_percent": float(row["pen_flourishing_percent"]),
            }
            for _, row in df.iterrows()
        ]
        existing_train = [(p, m) for p, m in zip(train_paths, train_meta) if p.exists()]
        if existing_train:
            paths, metas = zip(*existing_train)
            embs = embed_patches(model, list(paths), device)
            store_embeddings(collection, list(paths), embs, list(metas))
            total += len(paths)
            print(f"Training patches embedded: {len(paths)}")

    # ── Uploaded patches (flat directory, no metadata CSV) ──────────────────
    if UPLOADS_DIR.exists():
        upload_paths = sorted(UPLOADS_DIR.glob("*.png"))
        if upload_paths:
            # Preserve existing ChromaDB metadata (source_image_id, group, etc.)
            # by fetching it before upsert; fall back to a stub for brand-new entries.
            existing_docs = collection.get(
                ids=[p.name for p in upload_paths],
                include=["metadatas"],
            )
            meta_by_id = {
                doc_id: meta
                for doc_id, meta in zip(existing_docs["ids"], existing_docs["metadatas"])
                if meta is not None
            }
            upload_meta = [
                meta_by_id.get(p.name, {"source_image_id": -1, "group": "", "codex": ""})
                for p in upload_paths
            ]
            embs = embed_patches(model, upload_paths, device)
            store_embeddings(collection, upload_paths, embs, upload_meta)
            total += len(upload_paths)
            print(f"Uploaded patches embedded: {len(upload_paths)}")

    return total


def main():
    parser = argparse.ArgumentParser(description="Embed all patches into ChromaDB")
    parser.add_argument(
        "--collection",
        type=str,
        required=True,
        help="ChromaDB collection name (e.g. 'patches_v1'). Use distinct names to avoid overwriting.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(MODEL_PATH),
        help=f"Path to model checkpoint (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--patches_dir",
        type=str,
        default=str(PATCHES_DIR),
        help=f"Directory of extracted patch PNGs (default: {PATCHES_DIR})",
    )
    parser.add_argument(
        "--mlflow_run_id",
        type=str,
        default=None,
        help="MLflow run ID that produced the model checkpoint (stored in the ChromaDB collection metadata for traceability).",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = load_model(Path(args.model), device)

    import pandas as pd
    df = pd.read_csv(METADATA_CSV)
    patches_dir = Path(args.patches_dir)
    patch_paths = [patches_dir / row["patch_filename"] for _, row in df.iterrows()]
    metadatas = [
        {
            "source_image":            row["source_image"],
            "source_image_id":         int(row["source_image_id"]),
            "group":                   row["group"],
            "codex":                   row["codex"],
            "x":                       int(row["x"]),
            "y":                       int(row["y"]),
            "pen_flourishing_percent": float(row["pen_flourishing_percent"]),
        }
        for _, row in df.iterrows()
    ]

    print(f"Patches to embed: {len(patch_paths)}")
    print(f"Target collection: '{args.collection}'")

    embeddings = embed_patches(model, patch_paths, device)

    client = get_chroma_client(CHROMA_PATH)

    # Store mlflow_run_id in the collection metadata so every collection is
    # traceable back to the exact training run (weights, hyperparams, loss curves)
    # that produced it. Retrieve run details with: mlflow ui --backend-store-uri file://data/mlruns
    collection_metadata = {"mlflow_run_id": args.mlflow_run_id or "unknown"}
    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine", **collection_metadata},
    )

    store_embeddings(collection, patch_paths, embeddings, metadatas)
    print(f"Linked to MLflow run: {args.mlflow_run_id or 'not provided'}")


if __name__ == "__main__":
    main()
