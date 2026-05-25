import pathlib

import numpy as np
import torch
from PIL import Image

# Increases the sharpen focus (So narrows the region where the heatmap outputs orang/red)
HEATMAP_GAMMA = 2.0 


def _jet(h: np.ndarray) -> np.ndarray:
    """Map a [0,1] float32 array to uint8 RGB using the jet colormap."""
    r = np.clip(1.5 - np.abs(4.0 * h - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * h - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * h - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def save_heatmap_overlay(
    patch_image: Image.Image,
    heatmap: torch.Tensor,
    output_path: pathlib.Path,
    alpha: float = 0.5,
) -> None:
    h = heatmap.detach().cpu().numpy() if isinstance(heatmap, torch.Tensor) else np.asarray(heatmap)
    h = np.squeeze(h)
    if h.ndim != 2:
        raise ValueError(f"Expected 2D heatmap after squeeze, got shape {h.shape}")

    h = h.astype(np.float32)
    h = (h - h.min()) / (h.max() - h.min() + 1e-8)
    h = h ** HEATMAP_GAMMA

    colored = _jet(h)

    if patch_image.mode != "RGB":
        patch_image = patch_image.convert("RGB")

    colored_img = Image.fromarray(colored).resize(patch_image.size, Image.BILINEAR)

    patch_arr = np.asarray(patch_image, dtype=np.float32)
    heat_arr = np.asarray(colored_img, dtype=np.float32)
    blended = ((1.0 - alpha) * patch_arr + alpha * heat_arr).clip(0, 255).astype(np.uint8)

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(blended).save(output_path)
