import bisect
import json
import os
import random
from typing import Callable, Literal, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from services.ML.app.services.segment import (
    compute_patches,
    compute_mask_filter,
    pad_if_needed,
    get_codex,
    get_mask_name,
    is_image_file,
)


class SiameseScribeDataset(Dataset):
    """
    PyTorch Dataset for pen flourishing patches.

    Mirrors the PenFlourishingDataset pattern from the sample repository,
    adapted to the SiameseScribe folder structure. Extracts patches
    on-the-fly from the raw images using mask-based filtering.

    Returns an 8-tuple per patch:
        (crop, mask_patch, target, img_name, (x, y), pen_flourishing_percent, target, image_name)
    """

    def __init__(
        self,
        dataset_folder: str,
        patch_step_size: int,
        transformation_function: Optional[Callable],
        patch_size: Tuple[int, int],
        patch_filter_threshold: float,
        seed: int,
        mode: Literal["train", "test"],
    ) -> None:
        """
        :param dataset_folder: Root folder of the dataset (contains preprocessed/ and masks/).
        :param patch_step_size: Sliding window stride for patch generation.
        :param transformation_function: Optional transform applied to crop and mask patches.
        :param patch_size: Patch size as (height, width).
        :param patch_filter_threshold: Minimum pen flourishing coverage ratio to keep a patch.
        :param seed: Random seed for shuffling image order.
        :param mode: "train" or "test" — selects which preprocessed subfolder to use.
        """
        self.dataset_folder = os.path.join(dataset_folder, "preprocessed", mode)
        self.mask_root = os.path.join(dataset_folder, "masks")
        self.transformation_function = transformation_function
        self.patch_size = patch_size
        self.threshold = patch_filter_threshold
        self.seed = seed
        self.step_size = patch_step_size

        gt_path = os.path.join(self.dataset_folder, "ground_truth.txt")
        with open(gt_path, "r", encoding="utf-8") as f:
            ground_truth: dict = json.load(f)

        # Build label → integer index map (alphabetical for reproducibility: B=0, D=1, E=2, G=3)
        unique_labels = sorted(set(ground_truth.values()))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

        self.image_paths: list[tuple[str, int]] = []
        for group in sorted(os.listdir(self.dataset_folder)):
            group_dir = os.path.join(self.dataset_folder, group)
            if not os.path.isdir(group_dir):
                continue
            for root_dir, _, fnames in sorted(os.walk(group_dir, followlinks=True)):
                for fname in sorted(fnames):
                    if not is_image_file(fname):
                        continue
                    path = os.path.join(root_dir, fname)
                    label = ground_truth.get(fname)
                    if label is None:
                        continue
                    self.image_paths.append((path, self.label_to_idx[label]))

        random.seed(self.seed)
        random.shuffle(self.image_paths)

        self.image_metadata: list[dict] = []
        self.cumulative_counts: list[int] = [0]

        for path, target in tqdm(self.image_paths, desc=f"Indexing {mode} patches"):
            group = os.path.basename(os.path.dirname(path))
            image_name = os.path.basename(path)
            codex = get_codex(path)
            mask_name = get_mask_name(image_name)
            mask_path = os.path.join(self.mask_root, codex, group, mask_name)

            image = Image.open(path).convert("RGB")
            mask = Image.open(mask_path)

            image = pad_if_needed(image, self.patch_size)
            mask = pad_if_needed(mask, self.patch_size)

            # Single patchify call on the mask; positions are identical for image and mask
            mask_patches, positions = compute_patches(mask, "L", self.patch_size, self.step_size)
            valid_indices, valid_scores = compute_mask_filter(mask_patches, self.threshold)

            valid_positions = [positions[i] for i in valid_indices]

            self.image_metadata.append({
                "path": path,
                "target": target,
                "group": group,
                "image_name": image_name,
                "codex": codex,
                "mask_name": mask_name,
                "positions": valid_positions,
                "pen_flourishing_percents": valid_scores,
            })

            self.cumulative_counts.append(
                self.cumulative_counts[-1] + len(valid_positions)
            )

    def __len__(self) -> int:
        return self.cumulative_counts[-1]

    def __getitem__(self, idx: int) -> Tuple:
        img_idx = bisect.bisect_right(self.cumulative_counts, idx) - 1
        local_idx = idx - self.cumulative_counts[img_idx]
        meta = self.image_metadata[img_idx]

        image = Image.open(meta["path"]).convert("RGB")
        mask_path = os.path.join(
            self.mask_root, meta["codex"], meta["group"], meta["mask_name"]
        )
        mask = Image.open(mask_path)

        image = pad_if_needed(image, self.patch_size)
        mask = pad_if_needed(mask, self.patch_size)

        x, y = meta["positions"][local_idx]
        ph, pw = self.patch_size
        crop = image.crop((x, y, x + pw, y + ph))
        mask_patch = mask.crop((x, y, x + pw, y + ph))

        if self.transformation_function:
            crop = self.transformation_function(crop)
            mask_patch = self.transformation_function(mask_patch)

        img_name = meta["image_name"] + f"__patch{local_idx}"

        return (
            crop,
            mask_patch,
            meta["target"],
            img_name,
            (x, y),
            meta["pen_flourishing_percents"][local_idx],
            meta["target"],
            meta["image_name"],
        )
