"""
MLflow path portability helper.

MLflow stores absolute artifact_location in meta.yaml files when an experiment is first
created. If someone else created the experiment on a different machine, those paths point
to a non-existent directory on your machine and artifact logging fails.

Call fix_mlflow_paths() before mlflow.set_experiment() in any training script to repair
all experiment and run meta.yaml files to the current machine's project root.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[3]
sys.path.append(str(PROJECT_ROOT))

import yaml


def fix_mlflow_paths(mlflow_dir: Path, experiment_name: str) -> None:
    """
    Rewrite artifact_location / artifact_uri in MLflow meta.yaml files so they
    point to the current machine's mlruns directory instead of whoever created
    the experiment originally.

    Safe to call repeatedly — only rewrites files where the path is wrong.
    """
    if not mlflow_dir.exists():
        return

    for exp_dir in mlflow_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        meta_path = exp_dir / "meta.yaml"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = yaml.safe_load(f)

        if meta.get("name") != experiment_name:
            continue

        exp_id = meta["experiment_id"]
        correct_artifact_location = (mlflow_dir / exp_id).as_uri()

        if meta.get("artifact_location") != correct_artifact_location:
            meta["artifact_location"] = correct_artifact_location
            with open(meta_path, "w") as f:
                yaml.dump(meta, f, default_flow_style=False, allow_unicode=True)
            print(f"Fixed experiment artifact_location → {correct_artifact_location}")

        # Fix each run's artifact_uri
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue
            run_meta_path = run_dir / "meta.yaml"
            if not run_meta_path.exists():
                continue

            with open(run_meta_path) as f:
                run_meta = yaml.safe_load(f)

            run_id = run_meta.get("run_id", run_dir.name)
            correct_artifact_uri = (mlflow_dir / exp_id / run_id / "artifacts").as_uri()

            if run_meta.get("artifact_uri") != correct_artifact_uri:
                run_meta["artifact_uri"] = correct_artifact_uri
                with open(run_meta_path, "w") as f:
                    yaml.dump(run_meta, f, default_flow_style=False, allow_unicode=True)

        break
