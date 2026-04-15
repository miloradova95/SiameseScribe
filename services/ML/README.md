# ML Service — SiameseScribe

FastAPI service for patch extraction, embedding, similarity search, and model training/finetuning.

Runs on **port 8001**. Start from the repo root:

```bash
cd services/ML
uvicorn app.main:app --reload --port 8001
```

Interactive docs: http://localhost:8001/docs

---

## Status Overview

| Component | Status | Notes |
|---|---|---|
| Patch extraction (dataset) | Done | `extractDatasetPatches.py` — run once offline |
| `SiameseScribeDataset` | Done | On-the-fly PyTorch Dataset for training |
| `PatchTripletDataset` | Done | Triplet/pair dataset for metric learning |
| `SiameseNetwork` | Done | DenseNet121 backbone, 128-dim L2 embeddings |
| `TripletLoss` | Done | Standard triplet loss, margin=0.5 |
| `Training.py` | Done | Full training loop with MLflow tracking |
| `Embedd.py` | Done | Batch embed all patches into ChromaDB |
| `/embed_patches` endpoint | **Real** | Loads model, runs inference, returns 128-dim vectors |
| `/embed_all_patches` endpoint | Dummy | Stub — run `Embedd.py` directly instead |
| `/segment` endpoint | Dummy | Needs segmentation model for new images (not yet available) |
| `/search_patches` endpoint | Dummy | To be implemented after embedding is populated |
| `/explain_pair` endpoint | Dummy | SFAM heatmap generation — to be implemented |
| `/retrain` endpoint | Dummy | Fine-tuning on feedback — to be implemented |

---

## Prerequisites

Install dependencies from the repo root:

```bash
pip install -e .
pip install mlflow
```

---

## Step 1 — Extract Patches

Run once from the **repo root** to extract 128×128 patches from all manuscript images into
`data/patches/train/` and `data/patches/test/`, and generate the metadata CSVs.

```bash
python services/ML/app/services/extractDatasetPatches.py
```

Expected output:
```
[train] Found 630 images
  [1/630] CCl-71_020r0.jpg
  ...
[train] Done — XXXXX patches → data/patches/train
[train] Metadata written to data/patches/patches_train_metadata.csv

[test] Found 158 images
  ...
[test] Done — XXXXX patches → data/patches/test
[test] Metadata written to data/patches/patches_test_metadata.csv

All modes complete.
```

---

## Step 2 — Train the Model

Run from the **repo root**. Hyperparameters are defined as constants at the top of the file.

```bash
python services/ML/app/services/Training.py
```

Saves two files per run:
- `data/models/trainedModel_<run_id[:8]>.pth` — versioned copy, never overwritten
- `data/models/trainedModel.pth` — latest pointer, always updated (used by the FastAPI service)

Logs all runs to `data/mlruns/` (see MLflow section below).

Key config in `Training.py`:

| Constant | Default | Description |
|---|---|---|
| `EPOCHS` | 1 | Number of training epochs |
| `BATCH_SIZE` | 32 | Batch size |
| `LR` | 1e-4 | Learning rate (Adam) |
| `K_TRIPLETS` | 1 | Triplets generated per anchor per epoch |
| `EMBEDDING_DIM` | 128 | Embedding dimensionality |
| `MARGIN` | 0.5 | Triplet loss margin |

---

## Step 3 — Embed All Patches (populate ChromaDB)

Run after training. `--collection` is required — use a descriptive name so you can track
which collection corresponds to which model version.

```bash
python services/ML/app/services/Embedd.py  --collection patches_v1 --mlflow_run_id 9434e111de1e40bdb5a5ff1a4ce21822
```

The `--mlflow_run_id` links the ChromaDB collection back to the exact MLflow run that
produced the model weights, so you can always trace embeddings → weights → hyperparams.

Optional arguments:
```
--model       Path to .pth checkpoint  (default: data/models/trainedModel.pth)
--patches_dir Path to patch PNGs       (default: data/patches/train)
```

---

## MLflow — Experiment Tracking

All training runs are logged to `data/mlruns/`. Launch the UI from the repo root:

```bash
mlflow ui --backend-store-uri data/mlruns
```

Open: http://localhost:5000

Go to the `Evaluation runs` tab

Each run logs:
- **Parameters:** epochs, batch_size, lr, k_triplets, embedding_dim, margin, backbone, patch_size, device
- **Metrics:** `train_loss` per epoch
- **Artifacts:** model weights `.pth` file

---

## Testing the Live Endpoints

### Start the service

```bash
cd services/ML
uvicorn app.main:app --reload --port 8001
```

On startup the service loads the model from `data/models/trainedModel.pth`.
If no checkpoint exists yet it logs a warning and continues with random weights
(useful for testing endpoints before training finishes).

---

### Test `/embed_patches` (real)

```bash
curl -X POST http://localhost:8001/embed_patches \
  -H "Content-Type: application/json" \
  -d '{
    "patch_paths": [
      "data/patches/train/CCl-71_020r0.jpg__patch0.png",
      "data/patches/train/CCl-71_020r0.jpg__patch1.png"
    ]
  }'
```

Expected response — one 128-dim vector per patch:
```json
{
  "embeddings": [
    { "patch_path": "data/patches/train/CCl-71_020r0.jpg__patch0.png", "vector": [0.031, -0.012, ...] },
    { "patch_path": "data/patches/train/CCl-71_020r0.jpg__patch1.png", "vector": [0.018,  0.044, ...] }
  ]
}
```

Or use the interactive docs at http://localhost:8001/docs and paste any valid patch path
from `data/patches/train/`.

---

### Test `/embed_all_patches` (stub)

```bash
curl -X POST http://localhost:8001/embed_all_patches
```

Expected response:
```json
{
  "status": "started",
  "message": "Not yet implemented as a live endpoint. Run Embedd.py directly: ..."
}
```

---

### Test remaining stubs (`/segment`, `/search_patches`, `/explain_pair`, `/retrain`)

All return hardcoded mock responses. Use http://localhost:8001/docs to try them.

---

## Project Structure

```
services/ML/
├── app/
│   ├── main.py                        Entry point — loads model at startup
│   ├── routes/
│   │   └── api.py                     All ML endpoints
│   ├── services/
│   │   ├── segment.py                 Core patch extraction utilities (shared helpers)
│   │   ├── extractDatasetPatches.py   Offline script — extract patches from dataset
│   │   ├── SiameseScribeDataset.py    On-the-fly PyTorch Dataset (mirrors sample repo)
│   │   ├── PatchTripletDataset.py     Triplet/pair dataset for training
│   │   ├── SiameseNetwork.py          DenseNet121 siamese network
│   │   ├── TripletLoss.py             Triplet loss
│   │   ├── Training.py                Training script with MLflow
│   │   ├── Embedd.py                  Batch embedding script
│   │   ├── Old_Files/                 POC reference implementations
│   │   └── Sample_Repo_Files/         Supervisor's reference code
│   └── Endpoints+Services.md          API specification
└── README.md                          This file
```

---

## What Is Still Open

- **`/segment` for new images** — requires a segmentation model that generates masks for
  arbitrary uploaded images. The existing `segment.py` logic works but needs masks as input.
  Currently returns hardcoded mock patches.
- **`/search_patches`** — needs ChromaDB populated (run `Embedd.py` first).
- **`/explain_pair`** — SFAM heatmap generation using `SiameseNetwork.forward_with_sfam()`,
  needs implementing.
- **`/retrain`** — fine-tuning on user feedback using `TripletFeedbackDataset` (from Old_Files),
  needs adapting for patch paths.
