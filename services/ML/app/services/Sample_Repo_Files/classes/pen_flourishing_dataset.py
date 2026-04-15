from typing import Callable, Literal, Optional, Tuple

import bisect
import os
import random
from PIL import Image, ImageOps
import numpy as np
from torch.utils.data import Dataset

from tse_medieval.data_io.functions import compute_patches
from tse_medieval.utils import get_group_index_dict_from_folder, is_valid_img_file


class PenFlourishingDataset(Dataset):
    """
    Dataset that loads manuscript images, slices them into dense patches,
    applies mask-based filtering, and optionally computes saliency map weighting.
    """

    image_paths_storage = []

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
        Initialize dataset with dense patch extraction and mask filtering.

        :param dataset_folder: Root folder containing group subfolders with images.
        :type dataset_folder: str
        :param patch_step_size: Sliding window stride for patch generation.
        :type patch_step_size: int
        :param transformation_function: Optional transform applied to crops.
        :type transformation_function: Optional[callable]
        :param patch_size: Patch size (H, W).
        :type patch_size: tuple[int, int]
        :param patch_filter_threshold: Minimum pen flourishing foreground coverage ratio to keep a patch.
        :type patch_filter_threshold: float
        :param seed: Random seed for shuffling.
        :type seed: int
        :param mode: If the dataset is created for training or test data.
        :type mode: Literal["train", "test"]
        """
        self.dataset_folder = os.path.join(dataset_folder, "preprocessed", mode)
        self.mask_path = os.path.join(dataset_folder, "masks")
        self.transformation_function = transformation_function
        self.patch_size = patch_size
        self.threshold = patch_filter_threshold
        self.seed = seed
        self.step_size = patch_step_size

        directory, _, group_to_idx = get_group_index_dict_from_folder(
            data_dir=self.dataset_folder, log_indices=True
        )

        # Gather all image paths
        self.image_paths = []

        for target_group in sorted(group_to_idx.keys()):
            group_index = group_to_idx[target_group]
            target_dir = os.path.join(directory, target_group)
            if not os.path.isdir(target_dir):
                continue
            for root_dir, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root_dir, fname)
                    if is_valid_img_file(path):
                        self.image_paths.append((path, group_index))

        random.seed(self.seed)
        random.shuffle(self.image_paths)

        PenFlourishingDataset.image_paths_storage.extend(
            [img[0] for img in self.image_paths]
        )

        self.image_metadata = []
        self.cumulative_counts = [0]

        for path, target in self.image_paths:
            group = path.split("/")[-2]
            image_name = path.split("/")[-1]

            # Load and pad images and masks
            image = Image.open(path).convert("RGB")
            codex = "ccl73" if "CCl-73" in path else "ccl71"
            mask_name = image_name.replace("jpg", "png")
            mask_name = (
                mask_name.split("__")[0] + ".png" if "__" in mask_name else mask_name
            )

            mask_path = os.path.join(self.mask_path, codex, group, mask_name)
            mask = Image.open(mask_path)

            if image.size[0] < self.patch_size[0] or image.size[1] < self.patch_size[1]:
                height = (
                    self.patch_size[0]
                    if image.size[0] < self.patch_size[0]
                    else image.size[0]
                )
                width = (
                    self.patch_size[1]
                    if image.size[1] < self.patch_size[1]
                    else image.size[1]
                )

                image = ImageOps.pad(
                    image, (height, width), color=0, centering=(0.5, 0.5)
                )
                mask = ImageOps.pad(
                    mask, (height, width), color=0, centering=(0.5, 0.5)
                )

            # Generate mask patches and positions
            patch_size_img = (self.patch_size[0], self.patch_size[1], 3)
            patch_size_mask = self.patch_size

            _, positions = compute_patches(image, "RGB", patch_size_img, self.step_size)
            mask_patches, _ = compute_patches(
                mask, "L", patch_size_mask, self.step_size
            )

            # Filter by pen flourishing
            valid_positions, valid_pen_flourishing = [], []

            for n, m_patch in enumerate(mask_patches):
                pen_flourishing_percent = np.count_nonzero(m_patch == 255) / (
                    m_patch.shape[0] * m_patch.shape[1]
                )
                if pen_flourishing_percent >= self.threshold:
                    valid_positions.append(positions[n])
                    valid_pen_flourishing.append(pen_flourishing_percent)

            # Store metadata for this image
            self.image_metadata.append(
                {
                    "path": path,
                    "target": target,
                    "group": group,
                    "image_name": image_name,
                    "positions": valid_positions,
                    "pen_flourishing_percents": valid_pen_flourishing,
                }
            )

            # Update cumulative count
            self.cumulative_counts.append(
                self.cumulative_counts[-1] + len(valid_positions)
            )

    def __len__(self) -> int:
        """
        Number of patches in the dataset.

        :return: The length of the dataset.
        :rtype: int
        """
        return self.cumulative_counts[-1]

    def __getitem__(self, idx: int) -> Tuple:
        """
        Load single patch.

        :param idx: Index of the patch.
        :type idx: int
        :return: Single patch and corresponding metadata.
        :rtype: tuple
        """
        # Determine which image and which patch
        img_idx = bisect.bisect_right(self.cumulative_counts, idx) - 1
        local_idx = idx - self.cumulative_counts[img_idx]
        meta = self.image_metadata[img_idx]

        # Reload and pad
        image = Image.open(meta.get("path")).convert("RGB")
        codex = "ccl73" if "CCl-73" in meta.get("path") else "ccl71"
        mask_name = meta.get("image_name").replace("jpg", "png")
        mask_name = (
            mask_name.split("__")[0] + ".png" if "__" in mask_name else mask_name
        )
        current_item_mask_path = os.path.join(
            self.mask_path, codex, meta.get("group"), mask_name
        )
        mask = Image.open(current_item_mask_path)

        # Pad image if its smaller than the patch size
        if image.size[0] < self.patch_size[0] or image.size[1] < self.patch_size[1]:
            height = (
                self.patch_size[0]
                if image.size[0] < self.patch_size[0]
                else image.size[0]
            )
            width = (
                self.patch_size[1]
                if image.size[1] < self.patch_size[1]
                else image.size[1]
            )

            image = ImageOps.pad(image, (height, width), color=0, centering=(0.5, 0.5))
            mask = ImageOps.pad(mask, (height, width), color=0, centering=(0.5, 0.5))

        # Crop the selected patch
        x, y = meta.get("positions")[local_idx]
        ph, pw = self.patch_size
        crop = image.crop((x, y, x + pw, y + ph))
        mask_patch = mask.crop((x, y, x + pw, y + ph))

        # Apply transforms
        if self.transformation_function:
            crop = self.transformation_function(crop)
            current_mask = self.transformation_function(mask_patch)
        else:
            current_mask = mask_patch

        img_name = meta.get("image_name") + f"__patch{local_idx}"

        return (
            crop,
            current_mask,
            meta.get("target"),
            img_name,
            (x, y),
            meta.get("pen_flourishing_percents")[local_idx],
            meta.get("target"),
            meta.get("image_name"),
        )
