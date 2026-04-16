# SiameseScribe
An AI-Powered Morphological Retrieval Tool for Art History. Using Siamese Networks and high-dimensional vector embeddings to identify stylistic parallels in medieval pen flourishes. Features an expert-guided refinement loop where researchers evaluate similarity predictions to dynamically optimize model precision.

## Setup

### 1. Create virtual environment

Not yet added

---

### 2. Install dependencies

- 

---

### 3. Install project as package

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

If you have multiple Python installs on Windows, prefer `python -m uvicorn ...` over plain
`uvicorn ...` so the server starts with the same interpreter where you installed dependencies.
---

## Running the ML Backend

```bash
cd services/ml
python -m uvicorn app.main:app --reload --port 8001
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



## SQLite User Database

The project now includes a lightweight SQLite-backed user repository at `data/sqlite/userDatabase.py`.

It automatically creates `data/sqlite/users.sqlite3` and a `users` table with:
- unique `username`
- unique `email`
- `password_hash` stored with `bcrypt`
- timestamps for creation, update, and last login

Example usage:

```python
from data.sqlite import createUserDatabase

db = createUserDatabase()
user = db.createUser("alice", "alice@example.com", "super-secret")
authenticated = db.authenticateUser("alice", "super-secret")
allUsers = db.getAllUsers()
```

Run the smoke test with:

```powershell
python data/sqlite/testUserDatabase.py
```

Print all existing users with:

```powershell
python data/sqlite/showAllUsers.py
```
