import csv
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[3]

sys.path.append(str(PROJECT_ROOT))

from services.ML.app.services.segment import extract_patches, get_codex, get_mask_name, is_image_file

DATA_ROOT = PROJECT_ROOT / "data" / "dataset"
MASK_ROOT = DATA_ROOT / "masks"
PATCHES_ROOT = PROJECT_ROOT / "data" / "patches"

PATCH_SIZE = (128, 128)
STEP_SIZE = 64
THRESHOLD = 0.1
MODES = ["train", "test"]


def find_mask_path(image_path: Path) -> Path | None:
    codex = get_codex(image_path)
    group = image_path.parent.name
    mask_name = get_mask_name(image_path.name)
    mask_path = MASK_ROOT / codex / group / mask_name
    return mask_path if mask_path.exists() else None


def main():
    for mode in MODES:
        image_root = DATA_ROOT / "preprocessed" / mode
        output_dir = PATCHES_ROOT / mode

        if not image_root.exists():
            print(f"[{mode}] Image root not found: {image_root}, skipping.")
            continue

        all_images = [p for p in image_root.rglob("*") if is_image_file(p)]
        print(f"\n[{mode}] Found {len(all_images)} images")

        os.makedirs(output_dir, exist_ok=True)

        total_patches = 0
        metadata_rows = []

        for i, image_path in enumerate(all_images):
            print(f"  [{i+1}/{len(all_images)}] {image_path.name}")

            mask_path = find_mask_path(image_path)
            group = image_path.parent.name
            codex = get_codex(image_path)

            patches = extract_patches(
                image_path=str(image_path),
                patch_size=PATCH_SIZE,
                step_size=STEP_SIZE,
                output_dir=str(output_dir),
                mask_path=str(mask_path) if mask_path else None,
                threshold=THRESHOLD,
            )

            for patch in patches:
                x, y = patch["bbox"][0], patch["bbox"][1]
                metadata_rows.append({
                    "patch_filename": os.path.basename(patch["patch_path"]),
                    "source_image": image_path.name,
                    "group": group,
                    "codex": codex,
                    "x": x,
                    "y": y,
                    "pen_flourishing_percent": patch["score"],
                })

            total_patches += len(patches)

        csv_path = PATCHES_ROOT / f"patches_{mode}_metadata.csv"
        fieldnames = ["patch_filename", "source_image", "group", "codex", "x", "y", "pen_flourishing_percent"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metadata_rows)

        print(f"[{mode}] Done — {total_patches} patches → {output_dir}")
        print(f"[{mode}] Metadata written to {csv_path}")

    print("\nAll modes complete.")


if __name__ == "__main__":
    main()
