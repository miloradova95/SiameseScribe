# SiameseScribe
An AI-Powered Morphological Retrieval Tool for Art History. Using Siamese Networks and high-dimensional vector embeddings to identify stylistic parallels in medieval pen flourishes. Features an expert-guided refinement loop where researchers evaluate similarity predictions to dynamically optimize model precision.

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
