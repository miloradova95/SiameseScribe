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
from services.ML.app.services.ContrastiveLoss import ContrastiveLoss
from services.ML.app.services.mlflow_utils import fix_mlflow_paths

MODEL_PATH    = PROJECT_ROOT / "data" / "models" / "trainedModel.pth"
MLFLOW_DIR    = PROJECT_ROOT / "data" / "mlruns"
EMBEDDING_DIM = 128

EPOCHS     = 1
BATCH_SIZE = 16
LR         = 1e-4

# Strategy constants (see services/ML/Finetuning_Strategy.md)
W_AUG            = 0.7   # weight for augmented-anchor triplets
ALPHA_PAIR       = 0.2   # weight for positive-only pair loss
MAX_POS_NEG_RATIO = 2.0  # cap: effective_positive / effective_negative

_base_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Mild augmentation for anchor-as-positive (Type 2 samples).
# Preserves stroke structure — see strategy doc for rationale.
_aug_transforms = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.92, 1.00), ratio=(0.95, 1.05)),
    transforms.RandomRotation(degrees=5),
    transforms.RandomAffine(degrees=0, translate=(0.03, 0.03)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _resolve(path: str) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    # DB paths are relative to PROJECT_ROOT.parent ("SiameseScribe/data/…").
    # Smoke-test / segment paths are relative to PROJECT_ROOT ("data/…").
    # Try the DB convention first, fall back to the shorter form.
    candidate = PROJECT_ROOT.parent / resolved
    if candidate.exists():
        return candidate
    return PROJECT_ROOT / resolved


def _load(path: str) -> torch.Tensor:
    return _base_transforms(Image.open(_resolve(path)).convert("RGB"))


def _load_aug(path: str) -> torch.Tensor:
    return _aug_transforms(Image.open(_resolve(path)).convert("RGB"))


def _group_feedback(feedback: list[dict]) -> dict[str, dict[str, list[str]]]:
    groups: dict[str, dict[str, list[str]]] = {}
    for item in feedback:
        q = item["query_patch_path"]
        if q not in groups:
            groups[q] = {"pos": [], "neg": []}
        if item["is_similar"]:
            groups[q]["pos"].append(item["result_patch_path"])
        else:
            groups[q]["neg"].append(item["result_patch_path"])
    return groups


def _build_samples(
    feedback: list[dict],
    k_triplets: int,
) -> tuple[list[tuple[str, str, str, bool]], list[tuple[str, str]], int, int]:
    """
    Build all training samples from feedback.

    Returns:
        all_triplets  — (anchor, pos_path, neg_path, augment_pos) for real + aug triplets
        pos_pairs     — (anchor, pos_path) for positive-only anchors
        t_real        — count of complete user triplets (Type 1)
        t_aug         — count of augmented-anchor triplets (Type 2)
    """
    groups = _group_feedback(feedback)
    all_triplets: list[tuple[str, str, str, bool]] = []
    pos_pairs: list[tuple[str, str]] = []
    t_real = 0
    t_aug = 0

    for anchor, buckets in groups.items():
        pos_list = buckets["pos"]
        neg_list = buckets["neg"]
        has_pos = bool(pos_list)
        has_neg = bool(neg_list)

        if has_pos and has_neg:
            # Type 1 — complete user triplet
            for k in range(k_triplets):
                positive = random.choice(pos_list)
                negative = neg_list[k % len(neg_list)]
                all_triplets.append((anchor, positive, negative, False))
            t_real += 1

        elif has_neg:
            # Type 2 — neg-only: use augmented anchor as positive
            negative = random.choice(neg_list)
            all_triplets.append((anchor, anchor, negative, True))
            t_aug += 1

        elif has_pos:
            # Type 3 — pos-only: no reliable negative, use as pairwise pull
            for positive in pos_list:
                pos_pairs.append((anchor, positive))

    return all_triplets, pos_pairs, t_real, t_aug


def _apply_balance_cap(
    pos_pairs: list[tuple[str, str]],
    t_real: int,
    t_aug: int,
) -> list[tuple[str, str]]:
    """Subsample positive-only pairs so the pos/neg balance stays <= MAX_POS_NEG_RATIO."""
    effective_neg = t_real + W_AUG * t_aug
    if effective_neg == 0:
        return []
    max_pairs = int((MAX_POS_NEG_RATIO - 1.0) * effective_neg / ALPHA_PAIR)
    if len(pos_pairs) <= max_pairs:
        return pos_pairs
    return random.sample(pos_pairs, max_pairs)


# ─────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────

class _TripletDataset(Dataset):
    def __init__(self, triplets: list[tuple[str, str, str, bool]]):
        self.triplets = triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_path, pos_path, neg_path, augment_pos = self.triplets[idx]
        return (
            _load(anchor_path),
            _load_aug(pos_path) if augment_pos else _load(pos_path),
            _load(neg_path),
        )


class _PairDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx):
        anchor_path, pos_path = self.pairs[idx]
        return _load(anchor_path), _load(pos_path)


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def count_samples(feedback: list[dict]) -> dict:
    """
    Analyse feedback without building full samples.

    Returns {"t_real", "t_aug", "p_pos", "can_finetune"}.
    p_pos is already capped to the balance limit.
    can_finetune is True when t_real + t_aug >= 1.
    """
    groups = _group_feedback(feedback)
    t_real = 0
    t_aug = 0
    p_pos_raw = 0

    for buckets in groups.values():
        has_pos = bool(buckets["pos"])
        has_neg = bool(buckets["neg"])
        if has_pos and has_neg:
            t_real += 1
        elif has_neg:
            t_aug += 1
        elif has_pos:
            p_pos_raw += len(buckets["pos"])

    effective_neg = t_real + W_AUG * t_aug
    if effective_neg > 0:
        max_pairs = int((MAX_POS_NEG_RATIO - 1.0) * effective_neg / ALPHA_PAIR)
        p_pos = min(p_pos_raw, max_pairs)
    else:
        p_pos = 0

    return {
        "t_real": t_real,
        "t_aug": t_aug,
        "p_pos": p_pos,
        "can_finetune": (t_real + t_aug) >= 1,
    }


def finetune(feedback: list[dict], k_triplets: int = 1) -> dict:
    """
    Fine-tune the Siamese model on user feedback.

    Returns a dict: {run_id, triplets_used, t_real, t_aug, p_pos}.
    triplets_used counts real + aug triplets (not pairs).
    Returns run_id="" and zeros if the trigger condition is not met.
    """
    all_triplets, pos_pairs_raw, t_real, t_aug = _build_samples(feedback, k_triplets)

    if t_real + t_aug == 0:
        return {"run_id": "", "triplets_used": 0, "t_real": 0, "t_aug": 0, "p_pos": 0}

    pos_pairs = _apply_balance_cap(pos_pairs_raw, t_real, t_aug)
    p_pos = len(pos_pairs)

    balance_ratio = (
        (t_real + W_AUG * t_aug + ALPHA_PAIR * p_pos) / (t_real + W_AUG * t_aug)
        if (t_real + W_AUG * t_aug) > 0 else 1.0
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SiameseNetwork(embedding_dim=EMBEDDING_DIM).to(device)
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.train()
    model.feature_extractor.requires_grad_(False)

    triplet_criterion = TripletLoss()
    pair_criterion    = ContrastiveLoss()
    optimizer         = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )

    triplet_loader = DataLoader(
        _TripletDataset(all_triplets), batch_size=BATCH_SIZE, shuffle=True
    )
    pair_loader = (
        DataLoader(_PairDataset(pos_pairs), batch_size=BATCH_SIZE, shuffle=True)
        if pos_pairs else None
    )

    fix_mlflow_paths(MLFLOW_DIR, "siamese-scribe")
    mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
    mlflow.set_experiment("siamese-scribe")

    with mlflow.start_run() as run:
        run_id   = run.info.run_id
        short_id = run_id[:8]
        mlflow.set_tag("mlflow.runName", f"finetune_{short_id}")
        mlflow.log_params({
            "type":          "finetune",
            "epochs":        EPOCHS,
            "batch_size":    BATCH_SIZE,
            "lr":            LR,
            "k_triplets":    k_triplets,
            "feedback_items": len(feedback),
            "t_real":        t_real,
            "t_aug":         t_aug,
            "p_pos":         p_pos,
            "balance_ratio": round(balance_ratio, 4),
            "device":        device,
        })

        for epoch in range(EPOCHS):
            model.train()
            triplet_loss_sum = 0.0
            pair_loss_sum    = 0.0

            # Pass 1: triplet loss (real + augmented)
            pbar = tqdm(triplet_loader, desc=f"Finetune epoch {epoch + 1}/{EPOCHS} [triplets]", unit="batch")
            for anchor, positive, negative in pbar:
                anchor   = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)

                loss = triplet_criterion(
                    model.forward_once(anchor),
                    model.forward_once(positive),
                    model.forward_once(negative),
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                triplet_loss_sum += loss.item()
                pbar.set_postfix({"triplet_loss": f"{triplet_loss_sum / (pbar.n + 1):.4f}"})

            # Pass 2: pairwise pull loss (positive-only pairs)
            if pair_loader is not None:
                for anchor, positive in pair_loader:
                    anchor   = anchor.to(device)
                    positive = positive.to(device)

                    loss = ALPHA_PAIR * pair_criterion(
                        model.forward_once(anchor),
                        model.forward_once(positive),
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pair_loss_sum += loss.item()

            epoch_triplet_loss = triplet_loss_sum / len(triplet_loader)
            epoch_pair_loss    = pair_loss_sum / len(pair_loader) if pair_loader else 0.0

            mlflow.log_metric("finetune_triplet_loss", epoch_triplet_loss, step=epoch)
            if pair_loader:
                mlflow.log_metric("finetune_pair_loss", epoch_pair_loss, step=epoch)
            mlflow.log_metric(
                "finetune_total_loss",
                epoch_triplet_loss + ALPHA_PAIR * epoch_pair_loss,
                step=epoch,
            )

            print(
                f"Epoch {epoch + 1}/{EPOCHS}  "
                f"triplet_loss={epoch_triplet_loss:.4f}"
                + (f"  pair_loss={epoch_pair_loss:.4f}" if pair_loader else "")
            )

        versioned_path = MODEL_PATH.parent / f"trainedModel_ft_{short_id}.pth"
        torch.save(model.state_dict(), versioned_path)
        torch.save(model.state_dict(), MODEL_PATH)
        mlflow.log_artifact(str(versioned_path), artifact_path="weights")
        mlflow.set_tag("weights_file", versioned_path.name)

        print(f"Fine-tuned weights: {versioned_path}")
        print(f"MLflow run ID:      {run_id}")

    return {
        "run_id":        run_id,
        "triplets_used": len(all_triplets),
        "t_real":        t_real,
        "t_aug":         t_aug,
        "p_pos":         p_pos,
    }


if __name__ == "__main__":
    # Smoke test covering all three sample types
    sample_feedback = [
        # Anchor A: has both pos + neg → Type 1 real triplet
        {"query_patch_path": "data/patches/train/CCl-71_020r0.jpg__patch0.png",
         "result_patch_path": "data/patches/train/CCl-71_020r0.jpg__patch3.png",
         "is_similar": True},
        {"query_patch_path": "data/patches/train/CCl-71_020r0.jpg__patch0.png",
         "result_patch_path": "data/patches/train/CCl-71_020r0.jpg__patch1.png",
         "is_similar": False},
        # Anchor B: neg-only → Type 2 augmented triplet
        {"query_patch_path": "data/patches/train/CCl-71_020r0.jpg__patch5.png",
         "result_patch_path": "data/patches/train/CCl-71_020r0.jpg__patch6.png",
         "is_similar": False},
        # Anchor C: pos-only → Type 3 pairwise
        {"query_patch_path": "data/patches/train/CCl-71_020r0.jpg__patch9.png",
         "result_patch_path": "data/patches/train/CCl-71_020r0.jpg__patch10.png",
         "is_similar": True},
    ]
    stats = count_samples(sample_feedback)
    print(f"Sample counts: {stats}")
    result = finetune(sample_feedback, k_triplets=1)
    print(f"Done — {result}")
