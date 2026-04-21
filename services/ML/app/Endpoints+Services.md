# Pen Flourishing Analysis API Specification

## Overview

This document defines the REST API for the Pen Flourishing Medialab Project.

---

# Architecture Overview

- Frontend: Vue (JS)
- Main Backend: Fast API (state of the art, ML requests take longer on classical outing tools), Python
- ML Backend: FastAPI, Python, TrustAI Integration, Pytorch/Keras ... 
- Database: Chroma DB (specialized in similarity checks) 
- Storage(s): Local File Storage

![Component Overview](./Component_Overview.png)
---

# Services Overview

## Frontend (Vue)
- Upload images
- Display patches
- Trigger search
- Show results + explanations
- Send feedback

## Main Backend (FastAPI)
- Store images and patches
- Manage database
- Orchestrate ML calls
- Collect results for frontend

## ML Backend (FastAPI)
- Segment images
- Generate embeddings
- Perform similarity search
- Generate explanations
- Handle retraining

## Storage
- File system: images, patches, heatmaps
- Database: metadata, embeddings, feedback

---

# Sequence Flows

## Upload Flow:

User uploads image
=> Frontend => POST /images

Backend:
=> store image
=> call ML /segment
=> store patches
=> call ML /embed_patches
=> store embeddings

=> return patches to frontend


## Search Flow:
User clicks patch
=> Frontend => POST /patches/{id}/search

Backend:
=> get embedding from DB
=> call ML /search_patches
=> get similar patches

FOR EACH result:
=> call ML /explain_pair

=> return results + heatmaps

## Feedback Flow:

User clicks "Similar"
=> Frontend => POST /feedback

Backend:
=> store feedback
=> optionally trigger retrain
=> call ML /retrain

---

# MAIN BACKEND API

# 1. UPLOAD IMAGE

## POST /images

Uploads a full pen flourishing image and processes it into patches and embeddings.

---

### Request

Form Data:
file: <image file>

---

### Backend Processing Steps

1. Store image:
   - Save file to: `/data/images/{image_id}.png`
   - Create DB entry in `images`

2. Call ML Backend - segmentation:

   POST /segment
   {
     "image_path": "/data/images/{image_id}.png"
   }

3. Receive patches:
   - bbox
   - patch_path

4. Store patches in DB:
   - table: `patches`
   - fields: id, image_id, bbox, patch_path

5. Call ML Backend - embedding:

   POST /embed_patches
   {
     "patch_paths": [...]
   }

6. Store embeddings in DB

7. Return patches. 

---

### Response to the Frontend

{
  "image_id": 123,
  "patches": [
    {
      "patch_id": 5001,
      "bbox": [x, y, width, height]
    },
    {
      "patch_id": 5002,
      "bbox": [x, y, width, height]
    },
    {...}
  ]
}

---

# 2. SEARCH SIMILAR PATCHES

## POST /patches/{patch_id}/search

Search for similar patches based on a selected patch of the segmented input image. 

---

### Request

{
  "top_k": 4
}

---

### Backend Processing Steps

1. Fetch query patch:
   - DB lookup: `patches`
   - get `patch_path`, `image_id`

2. Fetch embedding:
   - DB lookup: `embeddings` (32-dim vector)

3. Call ML Backend - similarity search:

   POST /search_patches
   {
     "embedding": [32 floats],
     "top_k": 4
   }

4. Receive results:
   - patch_ids
   - similarity scores

5. For each result patch:
   - fetch metadata from DB:
     - image_id
     - bbox
     - patch_path

6. For EACH (query, result) pair:
   - Call ML Backend - Explanation heatmap

     POST /explain_pair
     {
       "query_patch_path": "...",
       "result_patch_path": "..."
     }

   - Receive:
     - explanation heatmap for query
     - explanation heatmap for result

7. Construct response:
   - include image URLs
   - include pairwise heatmaps

---

### Response

{
  "query_patch": {
    "patch_id": 5001,
    "image_id": 123,
    "image_url": "/images/123",
    "bbox": [...]
  },
  "results": [
    {
      "patch_id": 6001,
      "image_id": 200,
      "similarity_score": 0.91,

      "image_url": "/images/200",
      "patch_bbox": [...],

      "heatmaps": {
        "query": "/heatmaps/q5001_r6001.png",
        "result": "/heatmaps/r6001_q5001.png"
      }
    }
  ]
}

---

# 3. SUBMIT FEEDBACK

## POST /feedback

Stores user feedback on patch similarity. Base approach, feedback for similar or not similar.

---

### Request

{
  "query_patch_id": 5001,
  "result_patch_id": 6001,
  "label": "similar"
}

---

### Backend Processing Steps

1. Store feedback in DB:
   - table: `feedback`

2. (Optional) Trigger retraining on certain conditions:

   POST /retrain

---

### Response

{
  "status": "ok"
}

---

# 4. GET IMAGE

## GET /images/{image_id}

Returns full image.

---

### Backend Processing Steps

1. Lookup image in DB
2. Read file from storage
3. Return binary response

---

### Response

(binary image)

---

# 5. GET HEATMAP

## GET /heatmaps/{query_patch_id}/{result_patch_id}/{type}

Where:
- type = "query" or "result"

Example:
GET /heatmaps/5001/6001/query

---

### Backend Processing Steps

1. Construct file path:
   `/data/heatmaps/q{query}_r{result}.png`

2. Return file

---

### Response

(binary image)

---

# 6. OPTIONAL: RETRAIN TRIGGER

## POST /retrain

Triggers model fine-tuning manually based on collected user feedback.

---

### Backend Processing Steps

1. Fetch feedback from DB (table: `feedback`)
2. For each feedback row, resolve `query_patch_id` and `result_patch_id` to file paths
   via `patches.file_path`
3. Call ML backend with raw labeled pairs — triplet construction is done by ML:

   POST /retrain
   {
     "feedback": [
       {
         "query_patch_path": "data/patches/uploads/abc.jpg__patch0.png",
         "result_patch_path": "data/patches/train/CCl-71_020r0.jpg__patch3.png",
         "is_similar": true
       },
       {
         "query_patch_path": "data/patches/uploads/abc.jpg__patch0.png",
         "result_patch_path": "data/patches/train/CCl-71_020r0.jpg__patch1.png",
         "is_similar": false
       }
     ],
     "k_triplets": 1
   }

---

### Response

{
  "status": "training_started",
  "triplets_used": 1
}

Note: `triplets_used: 0` means feedback was received but no complete triplets could be
constructed (e.g. a query has only positive or only negative feedback — both are needed
to form a triplet).

<!-- ######################################################################## -->

# ML BACKEND API

# 1. SEGMENT IMAGE

## POST /segment

Splits a full image into patches using the segmentation model. Reuse the code and Model from Pea a fleu Gitlab for this task. 

---

### Request

{
  "image_path": "/data/images/{image_id}.png"
}

---

### Processing

- Load image from disk
- Run segmentation model
- Extract bounding boxes
- Save cropped patches to `/data/patches/`
- Generate patch file paths

---

### Response

{
  "patches": [
    {
      "patch_id": 5001,
      "bbox": [x, y, width, height],
      "patch_path": "/data/patches/5001.png"
    },
     {
      "patch_id": 5002,
      "bbox": [x, y, width, height],
      "patch_path": "/data/patches/5002.png"
    },
    {...}
  ]
}

---

# 2. EMBED PATCHES

## POST /embed_patches

Generates embeddings/feature vectors for patches. Returns them to the Backend which stores them in the DB. 

---

### Request

{
  "patch_paths": [
    "/data/patches/5001.png",
    "/data/patches/5002.png"
  ]
}

---

### Processing

- Load each patch
- Pass through the Siamese Network
- Extract feature vectors
- Pass through Encoder to reduce dimensionality


---

### Response

{
  "embeddings": [
    {
      "patch_path": "/data/patches/5001.png",
      "vector": [32 floats]
    }
  ]
}

---

# 3. SEARCH SIMILAR PATCHES

## POST /search_patches

Finds nearest neighbor patches using reduced embeddings.

---

### Request

{
  "embedding": [32 floats],
  "top_k": 4
}

---

### Processing

- Compare input embedding with stored embeddings
- Use Euclidean distance or cosine similarity
- Return closest matches

---

### Response

{
  "results": [
    {
      "patch_id": 6001,
      "similarity_score": 0.87
    },
    {
      "patch_id": 2134,
      "similarity_score": 0.87
    }
  ]
}

---

# 4. PAIRWISE HEATMAP

## POST /explain_pair

Generates explanation heatmaps for a pair of patches, highlighting the reasoning of the model behind their similarity. Unique heatmaps for each image pair. 

---

### Request

{
  "query_patch_path": "/data/patches/5001.png",
  "result_patch_path": "/data/patches/6001.png"
}

---

### Processing

1. Load both patches
2. Forward both through Siamese network
3. Compute similarity score for gradient attribution. (different similarity score then in search_patches, needed here to obtain gradients)
4. Backpropagate gradients from similarity score
5. Generate:
   - heatmap for query patch (w.r.t. result)
   - heatmap for result patch (w.r.t. query)
6. Save heatmaps to `/data/heatmaps/`

---

### Response

{
  "heatmaps": {
    "query": "/data/heatmaps/q5001_r6001.png",
    "result": "/data/heatmaps/r6001_q5001.png"
  }
}

---

# 5. RETRAIN MODEL

## POST /retrain

Fine-tunes the Siamese network on user feedback. Triplet construction is handled
internally by the ML service — the backend sends raw labeled pairs.

---

### Request

{
  "feedback": [
    {
      "query_patch_path": "data/patches/uploads/abc.jpg__patch0.png",
      "result_patch_path": "data/patches/train/CCl-71_020r0.jpg__patch3.png",
      "is_similar": true
    },
    {
      "query_patch_path": "data/patches/uploads/abc.jpg__patch0.png",
      "result_patch_path": "data/patches/train/CCl-71_020r0.jpg__patch1.png",
      "is_similar": false
    }
  ],
  "k_triplets": 1
}

- `feedback`: labeled pairs from user interactions (query patch + result patch + judgment)
- `k_triplets`: how many triplets to mine per anchor query (1–5, default 1)

---

### Processing

1. Group feedback by `query_patch_path`
2. For each query with at least one positive AND one negative result:
   - Construct (anchor, positive, negative) triplets
   - Rotate negatives for diversity when k_triplets > 1
3. Fine-tune Siamese network with Triplet Loss (LR=1e-6, 3 epochs)
4. Save versioned weights: `data/models/trainedModel_ft_{run_id[:8]}.pth`
5. Update latest pointer: `data/models/trainedModel.pth`
6. Log run to MLflow under experiment `siamese-scribe`

Training runs asynchronously — the response is returned immediately.

---

### Response

{
  "status": "training_started",
  "triplets_used": 1
}

- `triplets_used`: number of triplets constructed from the feedback.
  `0` means no complete triplets could be formed (queries need both positive
  and negative feedback to generate a triplet).


---

# NOTES

- Images and patches are stored in file storage (`data/`)
- ChromaDB stores embeddings; SQLite stores metadata and feedback
- Embeddings are 128-dim (DenseNet121 backbone, L2-normalised) — no autoencoder reduction
- Similarity metric: cosine similarity (`hnsw:space: cosine` in ChromaDB)
- System is patch-based, not full-image based
- Patch filenames follow the convention: `{source_image}__patch{index}.png`
- Patch paths in all API requests/responses are relative to the project root

---
