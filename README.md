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
