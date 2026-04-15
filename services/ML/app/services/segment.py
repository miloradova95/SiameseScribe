import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageOps
from patchify import patchify


def get_codex(path) -> str:
    return "ccl73" if "CCl-73" in str(path) else "ccl71"


def get_mask_name(image_name: str) -> str:
    mask_name = image_name.replace("jpg", "png")
    if "__" in mask_name:
        mask_name = mask_name.split("__")[0] + ".png"
    return mask_name


def is_image_file(path) -> bool:
    return Path(path).suffix.lower() in (".jpg", ".jpeg", ".png")


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
            positions.append((j * step_size, i * step_size))

    return flat_patches, positions


def compute_positions(image: Image.Image, patch_size: Tuple[int, int], step_size: int) -> list:
    """Compute patch positions without allocating patch arrays."""
    img_w, img_h = image.size
    ph, pw = patch_size
    positions = []
    for y in range(0, img_h - ph + 1, step_size):
        for x in range(0, img_w - pw + 1, step_size):
            positions.append((x, y))
    return positions


def pad_if_needed(image: Image.Image, patch_size: Tuple[int, int]) -> Image.Image:
    img_w, img_h = image.size  # PIL size is (width, height)
    target_w = max(img_w, patch_size[1])
    target_h = max(img_h, patch_size[0])

    if img_w < patch_size[1] or img_h < patch_size[0]:
        image = ImageOps.pad(image, (target_w, target_h), color=0, centering=(0.5, 0.5))

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


def extract_patches(
    image_path: str,
    patch_size: Tuple[int, int],
    step_size: int,
    output_dir: str,
    mask_path: str = None,
    threshold: float = 0.1,
):
    """Extract patches from image, optionally using mask filtering."""
    image = Image.open(image_path).convert("RGB")
    image = pad_if_needed(image, patch_size)

    if mask_path and os.path.exists(mask_path):
        mask = Image.open(mask_path).convert("L")
        mask = pad_if_needed(mask, patch_size)
        # Derive positions from mask patchify — avoids a redundant patchify on the image
        mask_patches, positions = compute_patches(mask, "L", patch_size, step_size)
        valid_indices, scores = compute_mask_filter(mask_patches, threshold)
    else:
        # Compute positions directly without allocating patch arrays
        positions = compute_positions(image, patch_size, step_size)
        valid_indices = list(range(len(positions)))
        scores = [None] * len(positions)

    os.makedirs(output_dir, exist_ok=True)

    patches = []
    ph, pw = patch_size

    for idx, patch_idx in enumerate(valid_indices):
        x, y = positions[patch_idx]
        crop = image.crop((x, y, x + pw, y + ph))

        patch_id = f"{os.path.basename(image_path)}__patch{idx}.png"
        patch_path = os.path.join(output_dir, patch_id)
        crop.save(patch_path)

        patches.append({
            "patch_id": idx,
            "bbox": [x, y, pw, ph],
            "patch_path": patch_path,
            "score": scores[idx],
        })

    return patches
