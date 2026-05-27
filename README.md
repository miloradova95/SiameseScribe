# SiameseScribe

An AI-powered morphological retrieval tool for art history. SiameseScribe uses Siamese networks and high-dimensional vector embeddings to identify stylistic parallels in medieval pen flourishes, with an expert-guided refinement loop for improving retrieval quality.

## Project Structure

- `services/ML` - ML API and retrieval logic
- `services/backend` - main backend API, database creation, and seeding
- `services/frontend` - website frontend
- `shared` - shared schemas and cross-service imports
- `data` - datasets, patches, models, and generated artifacts

## Prerequisites

- Python 3.11
- Conda
- Node.js 20.19.0 or newer
- npm
- Docker Desktop or Docker Engine with Docker Compose

## Setup

### 1. Create and activate the Python environment

```bash
conda create -n siamesescribe python=3.11
conda activate siamesescribe
```

### 2. Move into the project root

```bash
cd path/to/SiameseScribe
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

Installing the project in editable mode enables clean imports across services, for example:

```python
from shared.schemas.ml_backend import SegmentRequest
```

### 4. Install frontend dependencies

Run this once before starting the frontend, or anytime `package.json` / `package-lock.json` changes:

```bash
cd services/frontend
npm install
```

## Running with Docker

Start everything from the project root:

```bash
docker compose up --build
```

Run it in the background instead:

```bash
docker compose up --build -d
```

Open the app and APIs at:

```text
http://localhost          # frontend
http://localhost:8000/docs  # backend API
http://localhost:8001/docs  # ML service
http://localhost:8002/docs  # API gateway
```

Useful Docker commands:

```bash
docker compose ps
docker compose logs -f
docker compose down
```

The containers share the local `data/` directory through the `siamese-data` volume, so database files, models, patches, and other artifacts persist between restarts.

## Running the Project

Start each service in its own terminal from the project root.

### 1. Start the ML API

```bash
cd services/ML
uvicorn app.main:app --reload --port 8001
```

Available at:

```text
http://localhost:8001
http://localhost:8001/docs
```

### 2. Start the backend API

```bash
cd services/backend
python main.py
```

This creates and seeds the database on startup, then starts the backend API.

Available at:

```text
http://localhost:8000/docs
http://localhost:8000/images
http://localhost:8000/patches
```

### 3. Start the frontend

```bash
cd services/frontend
npm run dev
```

The frontend runs on Vite's local dev server, typically:

```text
http://localhost:5173
```

The frontend proxies `/api` requests to the backend at `http://localhost:8000`.

## Dataset Patch Extraction

Run this once from the project root before training if you need to extract `128x128` patches from the manuscript images:

```bash
python services/ML/app/services/extractDatasetPatches.py
```

This generates patch images and metadata under:

```text
data/patches/
```

The metadata CSV files map each patch back to its source image, group, codex, coordinates, and pen flourishing coverage score.

## Embedding Patches into ChromaDB

After training (or after restoring model weights), re-embed all patches so the ML service's similarity search reflects the current model. Run from the project root:

```bash
python -m services.ML.app.services.Embedd --collection patches_v1
```

This uses `data/models/trainedModel.pth` by default. To embed with a specific checkpoint:

```bash
python -m services.ML.app.services.Embedd --collection patches_v1 --model data/models/trainedModel_baseline.pth
```
