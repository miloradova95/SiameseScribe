# Pen Flourishing Segmentation

## Overview

Self-contained Streamlit application for automatic pen flourishing segmentation in medieval manuscript images using a trained U-Net++ deep learning model.

**Note:** This is a standalone application and is NOT part of the main PeuAFleu API service. The README previously mentioned image upload as a feature — this is fully implemented in `app.py` via Streamlit's file uploader. There is no REST API in this repo; the upload is browser-based only.

---

## What is actually in this repo

| File / Folder | Purpose |
|---|---|
| `app.py` | Streamlit web UI — upload JPEG(s), run segmentation, download mask(s) as PNG or ZIP |
| `segmentation_service.py` | Core service class (`SegmentationService`) — loads the model, runs inference, returns a mask. **No `__main__` block — running this file directly does nothing.** |
| `segmentation_utils/config.py` | Single dict `SEGMENTATION_CONFIG` with all inference parameters (patch size, step, confidence threshold, blob removal thresholds, default model path) |
| `segmentation_utils/segmentation_models.py` | PyTorch model definitions: `UNetV1` (vanilla U-Net), `UNetV2` (Attention U-Net), `UNetV3` (U-Net++ via SMP / ResNet34 encoder), DeepLabV3+, Mask R-CNN. `build_model_from_name()` picks the right one from the `.pth` filename prefix. |
| `segmentation_utils/prediction_utils.py` | All inference logic: sliding-window patching, per-patch forward pass, mask reconstruction, `blob_removal()` post-processing, and helper plotting functions |
| `segmentation_utils/eval_metrics.py` | Dice / IoU metrics and training-time evaluation helpers. **Not required for inference.** |
| `models/*.pth` | Two trained UNet-V3 (U-Net++) weights files |
| `requirements.txt` | Full dependency list (includes unrelated packages — see minimal list below) |
| `docker-compose.yml` / `Dockerfile` | Docker configuration — **not required** to run locally |

---

## How the segmentation pipeline works

```
Input JPEG/PNG bytes
        │
        ▼
1. PREPROCESSING
   - Open with PIL, convert to RGB numpy array (H×W×3)
   - Add black borders so H and W are evenly divisible
     by the patch grid (patch_size=512, step=341)

        │
        ▼
2. PATCH EXTRACTION  (patchify)
   - Sliding window of 512×512 patches, step 341
     (overlap of ~171 px per patch edge)

        │
        ▼
3. INFERENCE  (UNet-V3 / UNet++)
   - Each patch normalised to [0,1] float tensor
   - Forward pass → logits (1, 2, 512, 512)
   - softmax → argmax → per-pixel class (0=background, 1=flourish)
   - Foreground probability (class-1 softmax value) also recorded

        │
        ▼
4. RECONSTRUCTION
   - Patch binary votes accumulated into full-image count map
   - Any pixel with ≥1 positive vote → binary=1
   - Average probability map also built (grey map)

        │
        ▼
5. POSTPROCESSING  (blob_removal)
   - Connected-component analysis on the binary mask
   - Removes components where:
       area < 1000 px  OR  (width ≤ 50 AND height ≤ 50)
   - Surviving pixels set to 255, background to 0

        │
        ▼
Output: uint8 numpy array (H×W), values {0, 255}
        returned as PNG bytes or numpy array
```

---

## Why running `segmentation_service.py` directly does nothing

`segmentation_service.py` only **defines** the `SegmentationService` class and a `get_segmentation_service()` factory. There is no `if __name__ == "__main__":` block. The actual entry point is the Streamlit app.

---

## How to run locally (no Docker)

**Install the minimal required packages:**

```bash
pip install torch torchvision segmentation-models-pytorch==0.3.4        patchify==0.2.3 opencv-python pillow numpy streamlit==1.40.2
```

**Start the Streamlit app:**

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser, upload a JPEG manuscript image, select a model, click **Process Image**, and download the binary mask.

**To run inference programmatically (no browser):**

```python
from segmentation_service import get_segmentation_service

service = get_segmentation_service()  # loads default model

with open("my_image.jpg", "rb") as f:
    image_bytes = f.read()

mask_array, metadata = service.predict_mask(image_bytes)
# mask_array: uint8 numpy (H, W), values {0, 255}

png_bytes = service.mask_to_png_bytes(mask_array)
with open("mask_output.png", "wb") as f:
    f.write(png_bytes)
```

---

## Configuration

All inference parameters are in `segmentation_utils/config.py`:

| Key | Default | Meaning |
|---|---|---|
| `patch_size` | 512 | Patch size in pixels |
| `patch_step` | 341 | Sliding window step (512 − 512/3) |
| `confidence_threshold` | 0.8 | Used only for Mask R-CNN (ignored for UNet) |
| `num_classes` | 2 | Background + foreground |
| `blob_removal_min_area` | 1000 | Min pixel area to keep a component |
| `blob_removal_min_width` | 50 | Min width to keep a component |
| `blob_removal_min_height` | 50 | Min height to keep a component |
| `model_name` | `UNet-V3_28-11-2025_13-37.pth` | Default model file |
| `model_path` | `models/<model_name>` | Overridable via `SEGMENTATION_MODEL_PATH` env var |

---

## Model files

| File | Notes |
|---|---|
| `UNet-V3_28-11-2025_13-37.pth` | Most accurate; may miss some small flourishing |
| `UNet-V3_21-11-2025.pth` | Detects smaller flourishing better; more false positives |

Both are U-Net++ with a ResNet34 ImageNet encoder, 2 output classes.

---

## Integrating into another project — files to copy

To use the segmentation logic in a different project (without Streamlit, without Docker):

```
Copy these files/folders:
  segmentation_service.py          ← service class (your main interface)
  segmentation_utils/
    __init__.py
    config.py
    segmentation_models.py
    prediction_utils.py
    eval_metrics.py                ← only needed if you want Dice/IoU metrics
  models/
    UNet-V3_28-11-2025_13-37.pth  ← at minimum the default model
```

**Do NOT copy:** `app.py`, `Dockerfile`, `docker-compose.yml`, `.streamlit/`

**Add to your requirements:**

```
torch
torchvision
segmentation-models-pytorch==0.3.4
patchify==0.2.3
opencv-python
pillow
numpy
```

`streamlit` is **not** needed if you only use `SegmentationService` programmatically.

If `models/` lives in a different location in your project, either:
- Set the `SEGMENTATION_MODEL_PATH` env var to the absolute path of the `.pth` file, or
- Pass `model_name` to `SegmentationService` and adjust `config.py`'s `SEGMENTATION_DIR` accordingly.
