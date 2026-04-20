# -*- coding: utf-8 -*-
"""
Segmentation Service (Standalone)

Service for pen flourishing segmentation using trained U-Net model.
Handles image loading, prediction, and mask generation.

@date: 2025-12-16
"""

import io
import os
import sys
from pathlib import Path
from typing import Tuple

# Allow this module to be imported from anywhere — segmentation_utils is a sibling package
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import torch
from PIL import Image
from segmentation_utils.config import SEGMENTATION_CONFIG
from segmentation_utils.prediction_utils import blob_removal, get_image_prediction
from segmentation_utils.segmentation_models import build_model_from_name


class SegmentationService:
    """Service for handling pen flourishing segmentation operations."""

    def __init__(self, model_name=None):
        """Initialize the segmentation service and load the model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_type = None
        self.config = SEGMENTATION_CONFIG.copy()
        if model_name:
            model_path = Path(__file__).parent / "models" / model_name
            self.config["model_path"] = str(model_path)
            self.config["model_name"] = model_name
        self._load_model()

    def _load_model(self):
        """Load the segmentation model from disk."""
        try:
            model_path = self.config["model_path"]

            if not os.path.exists(model_path):
                print(f"ERROR: Model file not found: {model_path}")
                raise FileNotFoundError(f"Segmentation model not found at {model_path}")

            print(f"Loading segmentation model from: {model_path}")

            # Load state dict
            with open(model_path, "rb") as f:
                buffer = io.BytesIO(f.read())
            state_dict = torch.load(buffer, map_location=self.device)

            # Build model architecture
            model_name = self.config["model_name"]
            self.model, self.model_type = build_model_from_name(model_name, num_classes=self.config["num_classes"])

            # Load weights
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model.to(self.device)

            print(f"✓ Model loaded successfully: {model_name} (type: {self.model_type}, device: {self.device})")

        except Exception as e:
            print(f"ERROR: Failed to load segmentation model: {str(e)}")
            raise

    def predict_mask(self, image_bytes: bytes) -> Tuple[np.ndarray, dict]:
        """
        Predict segmentation mask for the given image.

        :param image_bytes: Raw image bytes (JPEG, PNG, etc.)
        :return: Tuple of (mask_array, metadata_dict)
            - mask_array: Binary mask as numpy array (H, W) with values {0, 255}
            - metadata_dict: Dictionary with prediction metadata
        """
        try:
            # Load image from bytes
            image = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Convert to numpy array
            image_rgb = np.array(image)

            print(f"Processing image: {image_rgb.shape}")

            # Get prediction
            binary_prediction, raw_prediction, gray_map = get_image_prediction(
                original_image_rgb=image_rgb,
                patch_size=self.config["patch_size"],
                step=self.config["patch_step"],
                device=self.device,
                model=self.model,
                confidence_thresh=self.config["confidence_threshold"],
                model_type=self.model_type,
            )

            # Apply blob removal (post-processing)
            # Convert to grayscale format for blob removal
            colored_pred = np.stack([binary_prediction * 255] * 3, axis=2).astype(np.uint8)
            mask_cleaned = blob_removal(
                colored_pred,
                min_area=self.config["blob_removal_min_area"],
                min_width=self.config["blob_removal_min_width"],
                min_height=self.config["blob_removal_min_height"],
            )

            # Prepare metadata
            metadata = {
                "image_shape": image_rgb.shape,
                "model_type": self.model_type,
                "patch_size": self.config["patch_size"],
                "patch_step": self.config["patch_step"],
                "confidence_threshold": self.config["confidence_threshold"],
                "device": str(self.device),
            }

            print(f"✓ Mask prediction completed: {mask_cleaned.shape}")

            return mask_cleaned, metadata

        except Exception as e:
            print(f"ERROR: Error during mask prediction: {str(e)}")
            raise

    def mask_to_png_bytes(self, mask: np.ndarray) -> bytes:
        """
        Convert mask array to PNG image bytes.

        :param mask: Binary mask as numpy array (H, W)
        :return: PNG image bytes
        """
        try:
            # Ensure mask is uint8
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)

            # Convert to PIL Image
            mask_image = Image.fromarray(mask, mode="L")

            # Save to bytes buffer
            buffer = io.BytesIO()
            mask_image.save(buffer, format="PNG")
            buffer.seek(0)

            return buffer.getvalue()

        except Exception as e:
            print(f"ERROR: Error converting mask to PNG: {str(e)}")
            raise


# Global instance
_segmentation_service = None


def get_segmentation_service(model_name=None) -> SegmentationService:
    """
    Get or create the global segmentation service instance.

    :param model_name: Optional model filename to load
    :return: SegmentationService instance
    """
    global _segmentation_service
    if _segmentation_service is None or (model_name and _segmentation_service.config["model_name"] != model_name):
        _segmentation_service = SegmentationService(model_name=model_name)
    return _segmentation_service
