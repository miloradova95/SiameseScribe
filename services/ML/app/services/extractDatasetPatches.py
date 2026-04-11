import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[3]

# Make imports work (shared, services, etc.)
sys.path.append(str(PROJECT_ROOT))

from services.ML.app.services.segment import extract_patches

DATA_ROOT = PROJECT_ROOT / "data" / "dataset"

IMAGE_ROOT = DATA_ROOT / "preprocessed" / "train"   # or "test"
MASK_ROOT = DATA_ROOT / "masks"

OUTPUT_ROOT = PROJECT_ROOT / "data" / "patches"

PATCH_SIZE = (128, 128)
STEP_SIZE = 64
THRESHOLD = 0.1


# DEBUG 


print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"IMAGE_ROOT: {IMAGE_ROOT}")
print(f"Exists IMAGE_ROOT: {IMAGE_ROOT.exists()}")
print(f"OUTPUT_ROOT: {OUTPUT_ROOT}")


# =========================
# HELPERS
# =========================

def find_mask_path(image_path: Path) -> Path | None:
    """
    Reconstruct mask path based on dataset logic
    """
    path_str = str(image_path)

    codex = "ccl73" if "CCl-73" in path_str else "ccl71"

    group = image_path.parent.name
    image_name = image_path.name

    mask_name = image_name.replace("jpg", "png")
    if "__" in mask_name:
        mask_name = mask_name.split("__")[0] + ".png"

    mask_path = MASK_ROOT / codex / group / mask_name

    return mask_path if mask_path.exists() else None


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in [".jpg", ".jpeg", ".png"]


# =========================
# MAIN SCRIPT
# =========================

def main():
    all_images = list(IMAGE_ROOT.rglob("*"))
    all_images = [p for p in all_images if is_image_file(p)]

    print(f"Found {len(all_images)} images")

    total_patches = 0

    for i, image_path in enumerate(all_images):
        print(f"[{i+1}/{len(all_images)}] Processing: {image_path.name}")

        mask_path = find_mask_path(image_path)

        patches = extract_patches(
            image_path=str(image_path),
            patch_size=PATCH_SIZE,
            step_size=STEP_SIZE,
            output_dir=str(OUTPUT_ROOT),
            mask_path=str(mask_path) if mask_path else None,
            threshold=THRESHOLD,
        )

        total_patches += len(patches)

    print(f"\nDone.")
    print(f"Total patches created: {total_patches}")


if __name__ == "__main__":
    main()