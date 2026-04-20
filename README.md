# SiameseScribe
An AI-Powered Morphological Retrieval Tool for Art History. Using Siamese Networks and high-dimensional vector embeddings to identify stylistic parallels in medieval pen flourishes. Features an expert-guided refinement loop where researchers evaluate similarity predictions to dynamically optimize model precision.

## Setup

## Setup Instructions

### 1. Create a new Conda environment (Python 3.11)

```bash
conda create -n siamesescribe python=3.11
conda activate siamesescribe
```

---

### 2. Navigate to the project directory

```bash
cd path/to/your/project
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```
---

### 4 Install project as package

Run this from the Root Directory:

```bash
pip install -e .
```

This enables clean imports across all services:
example in ml backend we can call. 

```python
from shared.schemas.ml_backend import SegmentRequest
```
And it imports from the shared folder automatically. 
---

## Running the ML Backend

```bash
cd services/ml
uvicorn app.main:app --reload --port 8001
```

Open:
```
http://localhost:8001/docs
```

---

## Extracting Dataset Patches (ML Preprocessing)

Run this once from the **root directory** before training to extract 128×128 patches from all manuscript images:

```bash
python services/ML/app/services/extractDatasetPatches.py
```

This processes both `train` and `test` splits and outputs to:

```
data/patches/
├── patches_train_metadata.csv
├── patches_test_metadata.csv
├── train/
│   └── *.png               (extracted patches)
└── test/
    └── *.png
```

The metadata CSV maps every patch back to its source image, group (B/D/E/G), codex, position (x, y), and pen flourishing coverage score. Only patches with ≥10% mask foreground coverage are kept.

---

## Running the Main Backend

```bash
cd services/backend
python main.py
```

Open:
```
http://localhost:8000/docs
```
Generating/populating the siamesescribe.db.
http://localhost:8000/images; http://localhost:8000/patches;
