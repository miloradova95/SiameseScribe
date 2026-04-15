import os
import random

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PatchTripletDataset(Dataset):
    """
    PyTorch Dataset for Siamese / metric learning on pre-extracted pen flourishing patches.

    Adapted from the POC SiameseDataset, replacing artist labels with rubricator groups (B/D/E/G)
    and loading patches directly from a flat PNG directory using the patches_metadata.csv index.

    Supports two modes:
        "triplet" → returns (anchor, positive, negative)
        "pair"    → returns (anchor, pair_image, label)  where label=1 (same group) / 0 (different)

    Pairs and triplets are generated dynamically — each call may return a different combination.
    When k_triplets > 1, each anchor image generates k triplets with diversity-guaranteed negatives
    (distinct rubricator groups across slots, seeded per anchor for reproducibility).
    """

    def __init__(
        self,
        csv_path: str,
        patches_dir: str,
        transform=None,
        balance: bool = True,
        mode: str = "triplet",
        k_triplets: int = 1,
    ):
        """
        :param csv_path: Path to patches_train_metadata.csv (columns: patch_filename, group, ...).
        :param patches_dir: Directory containing the extracted patch PNG files.
        :param transform: Torchvision transforms applied to each loaded patch.
        :param balance: If True, sample anchors uniformly across rubricator groups.
        :param mode: "triplet" or "pair".
        :param k_triplets: Number of triplets generated per anchor (triplet mode only).
        """
        self.patches_dir = patches_dir
        self.transform = transform
        self.balance = balance
        self.mode = mode
        self.k_triplets = max(1, k_triplets)

        df = pd.read_csv(csv_path)

        # Group patch filenames by rubricator group (B, D, E, G)
        self.group_to_patches: dict[str, list[str]] = {}
        for _, row in df.iterrows():
            group = row["group"]
            patch_filename = row["patch_filename"]
            if group not in self.group_to_patches:
                self.group_to_patches[group] = []
            self.group_to_patches[group].append(patch_filename)

        self.groups = list(self.group_to_patches.keys())
        self.all_patches = df["patch_filename"].tolist()

    def __len__(self):
        return len(self.all_patches) * self.k_triplets

    def __getitem__(self, idx):
        if self.mode == "triplet":
            return self._get_triplet(idx)
        return self._get_pair(idx)

    def _get_triplet(self, idx):
        """
        Returns (anchor, positive, negative).

        When k_triplets > 1:
          - anchor_idx pins the anchor across k slots
          - triplet_slot selects which of k negative groups to use
          - Negative groups are pre-shuffled per anchor_idx (seeded) for diversity
          - Positive is re-sampled fresh each call for variety
        """
        anchor_idx = idx // self.k_triplets
        triplet_slot = idx % self.k_triplets

        if self.balance:
            anchor_group = random.choice(self.groups)
            anchor_patch = random.choice(self.group_to_patches[anchor_group])
        else:
            anchor_patch = self.all_patches[anchor_idx]
            anchor_group = self._get_group(anchor_patch)

        anchor_img = self._load(anchor_patch)

        # Positive: same group, different patch
        positive_patch = anchor_patch
        while positive_patch == anchor_patch:
            positive_patch = random.choice(self.group_to_patches[anchor_group])
        positive_img = self._load(positive_patch)

        # Negative: different group, with diversity across k slots
        other_groups = [g for g in self.groups if g != anchor_group]

        if self.k_triplets > 1:
            rng = random.Random(anchor_idx)
            rng.shuffle(other_groups)
            negative_group = other_groups[triplet_slot % len(other_groups)]
        else:
            negative_group = random.choice(other_groups)

        negative_patch = random.choice(self.group_to_patches[negative_group])
        negative_img = self._load(negative_patch)

        return anchor_img, positive_img, negative_img

    def _get_pair(self, idx):
        """Returns (anchor, pair_image, label) where label=1 (same group) / 0 (different)."""
        anchor_idx = idx // self.k_triplets

        if self.balance:
            anchor_group = random.choice(self.groups)
            anchor_patch = random.choice(self.group_to_patches[anchor_group])
        else:
            anchor_patch = self.all_patches[anchor_idx]
            anchor_group = self._get_group(anchor_patch)

        anchor_img = self._load(anchor_patch)

        if random.random() < 0.5:
            pair_label = 1
            pair_patch = anchor_patch
            while pair_patch == anchor_patch:
                pair_patch = random.choice(self.group_to_patches[anchor_group])
        else:
            pair_label = 0
            other_groups = [g for g in self.groups if g != anchor_group]
            negative_group = random.choice(other_groups)
            pair_patch = random.choice(self.group_to_patches[negative_group])

        pair_img = self._load(pair_patch)
        return anchor_img, pair_img, pair_label

    def _load(self, patch_filename: str):
        path = os.path.join(self.patches_dir, patch_filename)
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def _get_group(self, patch_filename: str) -> str:
        """Reverse-lookup the group for a given patch filename."""
        for group, patches in self.group_to_patches.items():
            if patch_filename in patches:
                return group
        raise ValueError(f"Patch not found in any group: {patch_filename}")
