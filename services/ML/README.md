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
| `Training.py` | Done | Full training loop with MLflow tracking + per-epoch quick eval |
| `Embedd.py` | Done | Batch embed all patches into ChromaDB |
| `Evaluate.py` | Done | P@K and mAP eval against ChromaDB; logs to MLflow |
| `/embed_patches` endpoint | **Real** | Loads model, runs inference, returns 128-dim vectors |
| `/embed_all_patches` endpoint | Dummy | Stub — run `Embedd.py` directly instead |
| `/segment` endpoint | **Real** | U-Net++ segmentation → mask → 128×128 patches saved to `data/patches/uploads/` |
| `/search_patches` endpoint | **Real** | Queries ChromaDB with 128-dim vector, returns ranked filenames + cosine similarity scores |
| `/explain_pair` endpoint | Dummy | SFAM heatmap generation — to be implemented |
| `/retrain` endpoint | **Real** | Accepts feedback pairs, constructs triplets internally, fine-tunes in background |

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

Run after training from the **repo root**. Uses `data/models/trainedModel.pth` by default.

```bash
python -m services.ML.app.services.Embedd --collection patches_v1
```

Optional arguments:
```
--model           Path to a specific .pth checkpoint  (default: data/models/trainedModel.pth)
--patches_dir     Path to patch PNGs                  (default: data/patches/train)
--mlflow_run_id   Link this collection to a specific MLflow run for traceability
```

---

## Step 4 — Evaluate the Model

`Evaluate.py` measures retrieval quality using **Precision@K** and **mAP** (Mean Average Precision).
A retrieved patch is considered correct if its `group` label matches the query patch's group.

There are two evaluation modes:

### Full eval (standalone, ChromaDB-backed)

Runs after `Embedd.py` has populated ChromaDB with the current model's embeddings.
Queries all 25k test patches against the collection and computes definitive metrics.

```bash
python -m services.ML.app.services.Evaluate --collection patches_v1 --top_k 5
```

To log results directly onto an existing training run (eval metrics appear on the same run row in MLflow alongside `train_loss`):
```bash
python -m services.ML.app.services.Evaluate --collection patches_v1 --top_k 5 --mlflow_run_id 9434e111de1e40bdb5a5ff1a4ce21822
```

Expected output:
```
Device: cpu
Model loaded from data/models/trainedModel.pth
Collection 'patches_v1' loaded (90393 embeddings)
Embedding: 100%|████| 50/50 [00:XX<00:00]
Evaluating: 100%|████| 25603/25603 [00:XX<00:00]

=== Evaluation Results ===
Samples:       25603
Precision@5:   0.XXXX
mAP:           0.XXXX
Results saved to data/models/eval_results/<run_id>.json
```

The JSON artifact contains a full per-query breakdown (patch filename, group, P@K, AP) plus the aggregate summary.

Optional arguments:
```
--model       Path to .pth checkpoint  (default: data/models/trainedModel.pth)
--top_k       Number of results to evaluate against  (default: 5)
```

### Quick eval (per-epoch, automatic during training)

`Training.py` automatically runs a lightweight in-memory evaluation at the end of every epoch.
It samples ~1k gallery patches from the training set and ~1k query patches from the test set,
embeds both with the **current epoch's weights**, and computes P@K and mAP in memory — no ChromaDB needed.

This is used to:
- Track how retrieval quality improves across epochs in MLflow
- Save `trainedModel_best.pth` whenever a new best mAP is reached

No extra steps needed — just run training normally:
```bash
python services/ML/app/services/Training.py
```

Each epoch line will show:
```
Epoch 1/5  loss: 0.3241  P@5: 0.6120  mAP: 0.5834
Epoch 2/5  loss: 0.2987  P@5: 0.6450  mAP: 0.6102
...
Best epoch mAP: 0.6102  → trainedModel_best.pth
```

After training finishes, the script prints the exact commands to run next (embed → evaluate)
with the correct `--mlflow_run_id` already filled in.

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
- **Metrics:** `train_loss`, `eval/precision_at_k`, `eval/mAP` per epoch; `eval/precision_at_k` + `eval/mAP` added by `Evaluate.py` when `--mlflow_run_id` is passed
- **Artifacts:** model weights `.pth` file, eval results JSON (when full eval runs)

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

### Test `/segment` (real)

Requires the segmentation model weights at `services/ML/app/services/segmentation/models/UNet-V3_28-11-2025_13-37.pth`.

```bash
curl -X POST http://localhost:8001/segment \
  -H "Content-Type: application/json" \
  -d '{"image_path": "shared/schemas/TestImagesforUpload/CCl-71_017r0.jpg"}'
```

Expected response — one entry per extracted patch:
```json
{
  "patches": [
    {
      "patch_id": 0,
      "bbox": {"x": 64, "y": 128, "width": 128, "height": 128},
      "patch_path": "data/patches/uploads/CCl-71_017r0.jpg__patch0.png"
    },
    ...
  ]
}
```

Outputs written to disk:
- Mask PNG: `data/dataset/masks/uploads/CCl-71_017r0.png`
- Patches: `data/patches/uploads/CCl-71_017r0.jpg__patch{n}.png`

---

### Test `/search_patches` (real)

Requires ChromaDB to be populated first — run `Embedd.py` (Step 3 above).

Two-step process: embed a query patch, then search with the resulting vector.

**Step 1 — get an embedding:**
```bash
curl -X POST http://localhost:8001/embed_patches \
  -H "Content-Type: application/json" \
  -d '{"patch_paths": ["data/patches/train/CCl-71_020r0.jpg__patch0.png"]}'
```

Copy the `vector` array from the response (128 floats).

**Step 2 — search with it:**
```bash
curl -X POST http://localhost:8001/search_patches \
  -H "Content-Type: application/json" \
  -d '{
    "embedding": [0.031, -0.012, ...],
    "top_k": 5
  }'
```

Expected response:
```json
{
  "results": [
    { "patch_filename": "CCl-71_020r0.jpg__patch0.png", "similarity_score": 1.0 },
    { "patch_filename": "CCl-71_020r0.jpg__patch2.png", "similarity_score": 0.941 },
    { "patch_filename": "CCl-71_022r0.jpg__patch7.png", "similarity_score": 0.887 }
  ]
}
```

The top result should be the query patch itself (score ≈ 1.0). Results with the same group label (B/D/E/G) as the query should rank highest after a well-trained model.

---

### Test `/retrain` (real)

The endpoint returns immediately — training runs in the background. You need at least one query patch with **both** a positive and a negative feedback item to form a triplet.

```bash
curl -X POST http://localhost:8001/retrain \
  -H "Content-Type: application/json" \
  -d '{
    "feedback": [
      {
        "query_patch_path":  "data/patches/train/CCl-71_020r0.jpg__patch0.png",
        "result_patch_path": "data/patches/train/CCl-71_020r0.jpg__patch3.png",
        "is_similar": true
      },
      {
        "query_patch_path":  "data/patches/train/CCl-71_020r0.jpg__patch0.png",
        "result_patch_path": "data/patches/train/CCl-71_020r0.jpg__patch1.png",
        "is_similar": false
      }
    ],
    "k_triplets": 1
  }'
```

Expected response (immediate):
```json
{
  "status": "training_started",
  "triplets_used": 1
}
```

`triplets_used: 0` means no complete triplets could be formed — check that each query patch
has at least one `is_similar: true` **and** one `is_similar: false` result.

After training completes (background), check:
- `data/models/` — new `trainedModel_ft_{run_id}.pth` file
- MLflow UI (`mlflow ui --backend-store-uri data/mlruns`) — new `finetune_*` run with `finetune_loss` metric

---

### Test remaining stubs (`/explain_pair`)

Returns a hardcoded mock response. Use http://localhost:8001/docs to try it.

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
│   │   ├── Training.py                Training script with MLflow + per-epoch eval
│   │   ├── Embedd.py                  Batch embedding script
│   │   ├── Evaluate.py                P@K / mAP evaluation (full + quick modes)
│   │   ├── segmentation/              Segmentation service (U-Net++ model + utilities)
│   │   │   ├── segmentation_service.py    SegmentationService class — predict_mask()
│   │   │   ├── segmentation_utils/        Config, model definitions, prediction pipeline
│   │   │   └── models/                    Model weights (.pth files)
│   │   ├── Old_Files/                 POC reference implementations
│   │   └── Sample_Repo_Files/         Supervisor's reference code
│   └── Endpoints+Services.md          API specification
└── README.md                          This file
```

---

## What Is Still Open

- **`/explain_pair`** — SFAM heatmap generation using `SiameseNetwork.forward_with_sfam()`,
  needs implementing.
- **`/retrain`** — fine-tuning on user feedback using `TripletFeedbackDataset` (from Old_Files),
  needs adapting for patch paths.
  