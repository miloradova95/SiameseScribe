import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[3]
sys.path.append(str(PROJECT_ROOT))

import mlflow
import mlflow.pytorch
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from services.ML.app.services.SiameseNetwork import SiameseNetwork
from services.ML.app.services.TripletLoss import TripletLoss
from services.ML.app.services.PatchTripletDataset import PatchTripletDataset

# =========================
# CONFIG
# =========================

PATCHES_DIR     = PROJECT_ROOT / "data" / "patches" / "train"
METADATA_CSV    = PROJECT_ROOT / "data" / "patches" / "patches_train_metadata.csv"
MODEL_SAVE_PATH = PROJECT_ROOT / "data" / "models" / "trainedModel.pth"
MLFLOW_DIR      = PROJECT_ROOT / "data" / "mlruns"

EPOCHS        = 1
BATCH_SIZE    = 32
LR            = 1e-4
K_TRIPLETS    = 1
EMBEDDING_DIM = 128
MARGIN        = 0.5


# =========================
# TRANSFORMS
# =========================

def get_train_transforms():
    # Patches are already 128×128 — only normalize to ImageNet stats
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# =========================
# TRAINING
# =========================

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc="Training", unit="batch")
    for anchor, positive, negative in pbar:
        anchor   = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        emb_anchor   = model.forward_once(anchor)
        emb_positive = model.forward_once(positive)
        emb_negative = model.forward_once(negative)

        loss = criterion(emb_anchor, emb_positive, emb_negative)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{total_loss / (pbar.n + 1):.4f}"})

    return total_loss / len(dataloader)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dataset = PatchTripletDataset(
        csv_path=str(METADATA_CSV),
        patches_dir=str(PATCHES_DIR),
        transform=get_train_transforms(),
        balance=True,
        mode="triplet",
        k_triplets=K_TRIPLETS,
    )
    print(f"Dataset: {len(dataset)} triplets ({len(dataset.all_patches)} patches, "
          f"{len(dataset.groups)} groups: {sorted(dataset.groups)})")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model     = SiameseNetwork(embedding_dim=EMBEDDING_DIM).to(device)
    criterion = TripletLoss(margin=MARGIN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    os.makedirs(MODEL_SAVE_PATH.parent, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
    mlflow.set_experiment("siamese-scribe")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        short_id = run_id[:8]
        mlflow.set_tag("mlflow.runName", f"train_{short_id}")

        mlflow.log_params({
            "epochs":        EPOCHS,
            "batch_size":    BATCH_SIZE,
            "lr":            LR,
            "k_triplets":    K_TRIPLETS,
            "embedding_dim": EMBEDDING_DIM,
            "margin":        MARGIN,
            "backbone":      "densenet121",
            "patch_size":    128,
            "device":        device,
        })

        for epoch in range(EPOCHS):
            loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
            mlflow.log_metric("train_loss", loss, step=epoch)
            print(f"Epoch {epoch + 1}/{EPOCHS}  loss: {loss:.4f}")

        # Save a versioned copy (never overwritten) and update the latest pointer
        versioned_path = MODEL_SAVE_PATH.parent / f"trainedModel_{short_id}.pth"
        torch.save(model.state_dict(), versioned_path)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)  # latest pointer for the service

        # Log the versioned file into MLflow artifacts and tag it for visibility in the UI
        mlflow.log_artifact(str(versioned_path), artifact_path="weights")
        mlflow.set_tag("weights_file", versioned_path.name)

        print(f"\nVersioned weights: {versioned_path}")
        print(f"Latest pointer:    {MODEL_SAVE_PATH}")
        print(f"MLflow run ID:     {run_id}")
        print(f"View in UI:        mlflow ui --backend-store-uri \"{MLFLOW_DIR.as_uri()}\"")

    return run_id


if __name__ == "__main__":
    main()
