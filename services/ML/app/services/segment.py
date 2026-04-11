import os
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageOps
from patchify import patchify


# =========================
# CORE: PATCH EXTRACTION
# =========================

def compute_patches(
    image: Image.Image,
    color_mode: str,
    patch_size: Tuple[int, int],
    step_size: int,
):
    image_array = np.array(image)

    if color_mode == "RGB":
        patch_dims = (patch_size[0], patch_size[1], 3)
    else:
        patch_dims = (patch_size[0], patch_size[1])

    patches_grid = patchify(image_array, patch_dims, step=step_size)
    n_rows, n_cols = patches_grid.shape[0], patches_grid.shape[1]

    flat_patches = patches_grid.reshape(-1, *patch_dims)

    positions = []
    for i in range(n_rows):
        for j in range(n_cols):
            x = j * step_size
            y = i * step_size
            positions.append((x, y))

    return flat_patches, positions


# =========================
# HELPERS
# =========================

def pad_if_needed(image: Image.Image, patch_size: Tuple[int, int]) -> Image.Image:
    if image.size[0] < patch_size[0] or image.size[1] < patch_size[1]:
        height = max(image.size[0], patch_size[0])
        width = max(image.size[1], patch_size[1])

        image = ImageOps.pad(image, (height, width), color=0, centering=(0.5, 0.5))

    return image


def compute_mask_filter(mask_patches: np.ndarray, threshold: float):
    valid_indices = []
    valid_scores = []

    for i, m_patch in enumerate(mask_patches):
        percent = np.count_nonzero(m_patch == 255) / (
            m_patch.shape[0] * m_patch.shape[1]
        )

        if percent >= threshold:
            valid_indices.append(i)
            valid_scores.append(percent)

    return valid_indices, valid_scores


# =========================
# MAIN FUNCTION (WHAT YOU NEED)
# =========================

def extract_patches(
    image_path: str,
    patch_size: Tuple[int, int],
    step_size: int,
    output_dir: str,
    mask_path: str = None,
    threshold: float = 0.1,
):
    """
    Extract patches from image, optionally using mask filtering.
    """

    image = Image.open(image_path).convert("RGB")
    image = pad_if_needed(image, patch_size)

    # Compute image patches
    _, positions = compute_patches(image, "RGB", patch_size, step_size)

    # If mask exists → filter
    if mask_path and os.path.exists(mask_path):
        mask = Image.open(mask_path).convert("L")
        mask = pad_if_needed(mask, patch_size)

        mask_patches, _ = compute_patches(mask, "L", patch_size, step_size)

        valid_indices, scores = compute_mask_filter(mask_patches, threshold)
    else:
        # No filtering → keep all
        valid_indices = list(range(len(positions)))
        scores = [None] * len(positions)

    os.makedirs(output_dir, exist_ok=True)

    patches = []

    for idx, patch_idx in enumerate(valid_indices):
        x, y = positions[patch_idx]
        ph, pw = patch_size

        crop = image.crop((x, y, x + pw, y + ph))

        patch_id = f"{os.path.basename(image_path)}__patch{idx}.png"
        patch_path = os.path.join(output_dir, patch_id)

        crop.save(patch_path)

        patches.append(
            {
                "patch_id": idx,
                "bbox": [x, y, pw, ph],
                "patch_path": patch_path,
                "score": scores[idx],
            }
        )

    return patches