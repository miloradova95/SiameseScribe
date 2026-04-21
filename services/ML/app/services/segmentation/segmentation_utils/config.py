# -*- coding: utf-8 -*-
"""
Segmentation Configuration

Configuration settings for pen flourishing segmentation model.
"""

import os
from pathlib import Path

# Get the segmentation directory (where this file is located)
SEGMENTATION_DIR = Path(__file__).parent.parent

# Segmentation model configuration
SEGMENTATION_CONFIG = {
    # Model settings
    "confidence_threshold": 0.8,
    "num_classes": 2,  # background + foreground (fleuronnee)
    "patch_size": 512,
    "patch_step": 512 - int(512 / 3),  # overlapping patches step (for inference)
    # Model file path (self-contained within segmentation folder)
    "model_name": "UNet-V3_28-11-2025_13-37.pth",
    "model_path": os.getenv(
        "SEGMENTATION_MODEL_PATH", str(SEGMENTATION_DIR / "models" / "UNet-V3_28-11-2025_13-37.pth")
    ),
    # Blob removal settings (post-processing)
    "blob_removal_min_area": 1000,
    "blob_removal_min_width": 50,
    "blob_removal_min_height": 50,
}
