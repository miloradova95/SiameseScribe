import os
import random
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[3]
sys.path.append(str(PROJECT_ROOT))

import mlflow
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from services.ML.app.services.SiameseNetwork import SiameseNetwork
from services.ML.app.services.TripletLoss import TripletLoss
from services.ML.app.services.mlflow_utils import fix_mlflow_paths

MODEL_PATH   = PROJECT_ROOT / "data" / "models" / "trainedModel.pth"
MLFLOW_DIR   = PROJECT_ROOT / "data" / "mlruns"
EMBEDDING_DIM = 128

EPOCHS     = 3
BATCH_SIZE = 16
LR         = 1e-6
K_TRIPLETS_DEFAULT = 1

_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _build_triplets(feedback: list[dict], k_triplets: int) -> list[tuple[str, str, str]]:
    """Group feedback by query path and mine (anchor, positive, negative) triplets."""
    groups: dict[str, dict[str, list[str]]] = {}
    for item in feedback:
        q = item["query_patch_path"]
        if q not in groups:
            groups[q] = {"pos": [], "neg": []}
        if item["is_similar"]:
            groups[q]["pos"].append(item["result_patch_path"])
        else:
            groups[q]["neg"].append(item["result_patch_path"])

    triplets = []
    for anchor, buckets in groups.items():
        pos_list = buckets["pos"]
        neg_list = buckets["neg"]
        if not pos_list or not neg_list:
            continue
        for k in range(k_triplets):
            positive = random.choice(pos_list)
            negative = neg_list[k % len(neg_list)]
            triplets.append((anchor, positive, negative))

    return triplets


def count_constructable_triplets(feedback: list[dict], k_triplets: int) -> int:
    return len(_build_triplets(feedback, k_triplets))


class _TripletDataset(Dataset):
    def __init__(self, triplets: list[tuple[str, str, str]]):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_path, pos_path, neg_path = self.triplets[idx]

        def load(p):
            resolved = Path(p)
            if not resolved.is_absolute():
                resolved = PROJECT_ROOT / resolved
            return _transforms(Image.open(resolved).convert("RGB"))

        return load(anchor_path), load(pos_path), load(neg_path)


def finetune(feedback: list[dict], k_triplets: int = K_TRIPLETS_DEFAULT) -> tuple[str, int]:
    """Fine-tune the Siamese model on user feedback. Returns (mlflow_run_id, triplets_used)."""
    triplets = _build_triplets(feedback, k_triplets)
    if not triplets:
        return "", 0

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SiameseNetwork(embedding_dim=EMBEDDING_DIM).to(device)
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.train()

    criterion = TripletLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    dataset    = _TripletDataset(triplets)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    fix_mlflow_paths(MLFLOW_DIR, "siamese-scribe")
    mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
    mlflow.set_experiment("siamese-scribe")

    with mlflow.start_run() as run:
        run_id   = run.info.run_id
        short_id = run_id[:8]
        mlflow.set_tag("mlflow.runName", f"finetune_{short_id}")
        mlflow.log_params({
            "type":        "finetune",
            "epochs":      EPOCHS,
            "batch_size":  BATCH_SIZE,
            "lr":          LR,
            "k_triplets":  k_triplets,
            "feedback_items": len(feedback),
            "triplets":    len(triplets),
            "device":      device,
        })

        for epoch in range(EPOCHS):
            total_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Finetune epoch {epoch + 1}/{EPOCHS}", unit="batch")
            for anchor, positive, negative in pbar:
                anchor   = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)

                loss = criterion(
                    model.forward_once(anchor),
                    model.forward_once(positive),
                    model.forward_once(negative),
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{total_loss / (pbar.n + 1):.4f}"})

            epoch_loss = total_loss / len(dataloader)
            mlflow.log_metric("finetune_loss", epoch_loss, step=epoch)
            print(f"Epoch {epoch + 1}/{EPOCHS}  loss: {epoch_loss:.4f}")

        versioned_path = MODEL_PATH.parent / f"trainedModel_ft_{short_id}.pth"
        torch.save(model.state_dict(), versioned_path)
        torch.save(model.state_dict(), MODEL_PATH)
        mlflow.log_artifact(str(versioned_path), artifact_path="weights")
        mlflow.set_tag("weights_file", versioned_path.name)

        print(f"Fine-tuned weights: {versioned_path}")
        print(f"MLflow run ID:      {run_id}")

    return run_id, len(triplets)


if __name__ == "__main__":
    # Quick smoke test
    sample_feedback = [
        {"query_patch_path": "data/patches/train/CCl-71_020r0.jpg__patch0.png",
         "result_patch_path": "data/patches/train/CCl-71_020r0.jpg__patch3.png",
         "is_similar": True},
        {"query_patch_path": "data/patches/train/CCl-71_020r0.jpg__patch0.png",
         "result_patch_path": "data/patches/train/CCl-71_020r0.jpg__patch1.png",
         "is_similar": False},
    ]
    run_id, n = finetune(sample_feedback, k_triplets=1)
    print(f"Done — run_id={run_id}, triplets={n}")
