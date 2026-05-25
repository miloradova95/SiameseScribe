# Dataset Upload Guide

The `./data` directory is gitignored and not included in the Docker image.
On a fresh server deploy the data volume is empty — this guide covers how to push
the static dataset (train/test patches + source manuscript images) to the server.

Model files (`data/models/*.pth`) are committed to git and are already present after deploy.
Metadata CSVs (`patches_train_metadata.csv`, `patches_test_metadata.csv`) are also in git.

**Requirement:** `pip install httpx` (if not already installed).

---

## What should get uploaded

| Directory | Contents |
|---|---|
| `patches/train/` | ~90k 128×128 patch PNGs |
| `patches/test/` | ~25k 128×128 patch PNGs |
| `dataset/preprocessed/train/` | Source manuscript images (train split) |
| `dataset/preprocessed/test/` | Source manuscript images (test split) |

---

## 1. Check what's on the server

Always run this first to confirm you're hitting the right server and the data volume is mounted:

```bash
python scripts/upload_dataset_to_server.py \
  --server-url https://YOUR_SERVER_URL \
  --admin-password YOUR_ADMIN_PASSWORD \
  --info
```

---

## 2. Local test with a small batch

Before running the full upload, verify the endpoint works by uploading a few test files.
The test directory must live **outside** `./data/` — otherwise the bind mount makes those
files already visible to the container and nothing would be uploaded.

**Create test files:**
```bash
mkdir -p test_upload/patches/train

cp data/patches/train/CCl-71_020r0__patch0.png test_upload/patches/train/TEST_patch_001.png
cp data/patches/train/CCl-71_020r0__patch1.png test_upload/patches/train/TEST_patch_002.png
cp data/patches/train/CCl-71_020r0__patch2.png test_upload/patches/train/TEST_patch_003.png
```

**Dry run (no files sent):**
```bash
python scripts/upload_dataset_to_server.py \
  --server-url http://localhost \
  --admin-password changeme123 \
  --data-dir ./test_upload \
  --dirs patches/train \
  --dry-run
```

**Actual upload:**
```bash
python scripts/upload_dataset_to_server.py   --server-url http://localhost   --admin-password YOUR_ADMIN_PASSWORD   --data-dir ./test_upload   --dirs patches/train
```

Expected: 3 files uploaded.

**Confirm they landed (local Docker):**
```bash
docker exec siamesescribe-backend-1 ls /app/data/patches/train/TEST_patch_001.png
```

**Run again — should skip all 3 (idempotency check):**
```bash
python scripts/upload_dataset_to_server.py   --server-url http://localhost   --admin-password YOUR_ADMIN_PASSWORD   --data-dir ./test_upload   --dirs patches/train
```

Expected: `Files to upload: 0 — Nothing to upload — server is already up to date.`

**Clean up:**
```bash
rm data/patches/train/TEST_patch_00*.png
rm -rf test_upload/
```

---

## 3. Full upload to the production server

**Dry run first:**
```bash
python scripts/upload_dataset_to_server.py   --server-url https://YOUR_SERVER_URL   --admin-password YOUR_ADMIN_PASSWORD   --dry-run
```

**Full upload (all four directories):**
```bash
python scripts/upload_dataset_to_server.py   --server-url https://YOUR_SERVER_URL   --admin-password YOUR_ADMIN_PASSWORD
```

This will take a while (~115k files). Progress is printed per file.
The script is safe to re-run — it skips files already present with the correct size.

**Upload specific directories only:**
```bash
python scripts/upload_dataset_to_server.py \
  --server-url https://YOUR_SERVER_URL \
  --admin-password YOUR_ADMIN_PASSWORD \
  --dirs patches/train patches/test
```

---

## 4. After upload — next steps

Uploading files does **not** automatically populate ChromaDB (the vector store used for
similarity search). That requires a separate re-embedding step — to be documented once
the triggering mechanism is decided.

---

## Script reference

```
--server-url      Base URL of the server (e.g. https://example.com or http://localhost)
--admin-user      Admin username (default: admin)
--admin-password  Admin password
--data-dir        Local data/ directory to sync from (default: ./data)
--dirs            Limit to specific subdirectories (default: all four)
--concurrency     Parallel uploads (default: 10)
--dry-run         Show what would be uploaded, send nothing
--info            Show server directory state and exit (fast, no file counting)
--count-files     Add to --info to include file counts (slow on large directories)
--base-path       API prefix (default: /api). Use "" when hitting backend on port 8000 directly
```

---

## Troubleshooting

**504 Gateway Timeout on inventory:** The file walk over 115k files is slow.
nginx timeout is set to 300s — if it still times out, re-run; the server wasn't under load.

**`docker cp` workflow for local backend changes (avoids full rebuild):**
```bash
docker cp services/backend/routes/dataset_sync.py siamesescribe-backend-1:/app/services/backend/routes/dataset_sync.py
docker compose restart backend

# For nginx.conf changes:
docker cp services/frontend/nginx.conf siamesescribe-frontend-1:/etc/nginx/conf.d/default.conf
docker exec siamesescribe-frontend-1 nginx -s reload
```
