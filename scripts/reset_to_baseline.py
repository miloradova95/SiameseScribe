"""
reset_to_baseline.py — dev utility for rolling back to the base-trained model.

What it does:
  1. Restores data/models/trainedModel.pth from the baseline snapshot
  2. Deletes all fine-tuned weight files (trainedModel_ft_*.pth)
  3. Deletes all FinetuneRun records from the database
  4. Resets feedback.used_for_retrain = 0 so labels can be reused

After running: restart the ML service and re-embed (Embedd.py or /reembed_all).
"""

import shutil
import sqlite3
import sys
from pathlib import Path

REPO_ROOT      = Path(__file__).resolve().parents[1]
MODELS_DIR     = REPO_ROOT / "data" / "models"
BASELINE_PATH  = MODELS_DIR / "trainedModel_baseline.pth"
LIVE_PATH      = MODELS_DIR / "trainedModel.pth"
DB_PATH        = REPO_ROOT / "data" / "sqlite" / "siamesescribe.db"


def confirm(prompt: str) -> bool:
    answer = input(f"{prompt} [y/N] ").strip().lower()
    return answer == "y"


def main() -> None:
    print("=" * 60)
    print("SiameseScribe — baseline rollback script")
    print("=" * 60)
    print()
    print("This will:")
    print(f"  • Copy {BASELINE_PATH.name} → trainedModel.pth")
    print(f"  • Delete all  trainedModel_ft_*.pth  files in {MODELS_DIR.name}/")
    print(f"  • Delete ALL  FinetuneRun rows  from the database")
    print(f"  • Reset  feedback.used_for_retrain = 0  (labels kept)")
    print()

    if not BASELINE_PATH.exists():
        print(f"ERROR: Baseline weights not found at:\n  {BASELINE_PATH}")
        print("Cannot proceed.")
        sys.exit(1)

    if not DB_PATH.exists():
        print(f"ERROR: Database not found at:\n  {DB_PATH}")
        print("Cannot proceed.")
        sys.exit(1)

    if not confirm("Proceed?"):
        print("Aborted — nothing changed.")
        sys.exit(0)

    print()

    # 1. Restore baseline weights
    shutil.copy2(BASELINE_PATH, LIVE_PATH)
    print(f"[OK] Restored weights: {BASELINE_PATH.name} → trainedModel.pth")

    # 2. Remove fine-tuned weight files
    ft_files = list(MODELS_DIR.glob("trainedModel_ft_*.pth"))
    if ft_files:
        for f in ft_files:
            f.unlink()
            print(f"[OK] Deleted {f.name}")
    else:
        print("[--] No trainedModel_ft_*.pth files found")

    # 3 & 4. Database operations
    print()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()

        cur.execute("DELETE FROM finetune_runs")
        deleted_runs = cur.rowcount
        print(f"[OK] Deleted {deleted_runs} FinetuneRun record(s)")

        cur.execute("UPDATE feedback SET used_for_retrain = 0")
        reset_feedback = cur.rowcount
        print(f"[OK] Reset used_for_retrain on {reset_feedback} feedback record(s)")

        conn.commit()

    print()
    print("=" * 60)
    print("Done. Next steps:")
    print("  1. Restart the ML service (Ctrl+C then re-run uvicorn)")
    print("  2. Re-embed patches:")
    print("       python -m services.ML.app.services.Embedd --collection patches_v1")
    print("     or call POST /reembed_all via the admin UI / ML service Swagger")
    print("=" * 60)


if __name__ == "__main__":
    main()
