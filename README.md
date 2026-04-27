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

## Environment Variables

The backend requires a `.env` file in `services/backend/` for JWT auth and admin user seeding. Create it before starting the backend:

```bash
# services/backend/.env
SECRET_KEY=change-this-to-a-random-secret-before-deploying
ADMIN_USERNAME=admin
ADMIN_EMAIL=admin@siamesescribe.local
ADMIN_PASSWORD=changeme123
```

> Change `SECRET_KEY` and `ADMIN_PASSWORD` before any deployment.

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