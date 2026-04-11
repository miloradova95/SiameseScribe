from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import bcrypt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "sqlite" / "users.sqlite3"


@dataclass(frozen=True)
class UserRecord:
    id: int
    username: str
    email: str
    password_hash: str
    created_at: str
    updated_at: str
    last_login_at: Optional[str]


class UserDatabase:
    def __init__(self, dbPath: Path | str = DEFAULT_DB_PATH) -> None:
        self.dbPath = Path(dbPath)
        self.dbPath.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.dbPath)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with closing(self.connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_login_at TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS users_set_updated_at
                AFTER UPDATE ON users
                FOR EACH ROW
                BEGIN
                    UPDATE users
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE id = OLD.id;
                END
                """
            )
            connection.commit()

    @staticmethod
    def hashPassword(password: str) -> str:
        passwordBytes = password.encode("utf-8")
        return bcrypt.hashpw(passwordBytes, bcrypt.gensalt()).decode("utf-8")

    @staticmethod
    def verifyPassword(password: str, passwordHash: str) -> bool:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            passwordHash.encode("utf-8"),
        )

    def createUser(self, username: str, email: str, password: str) -> UserRecord:
        passwordHash = self.hashPassword(password)

        try:
            with closing(self.connect()) as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO users (username, email, password_hash)
                    VALUES (?, ?, ?)
                    """,
                    (username, email, passwordHash),
                )
                userId = cursor.lastrowid
                connection.commit()
        except sqlite3.IntegrityError as exc:
            raise ValueError("Username or email already exists.") from exc

        user = self.getUserById(userId)
        if user is None:
            raise RuntimeError("User was inserted but could not be loaded back.")
        return user

    def getUserById(self, userId: int) -> Optional[UserRecord]:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE id = ?",
                (userId,),
            ).fetchone()
        return self.rowToUser(row)

    def getUserByEmail(self, email: str) -> Optional[UserRecord]:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE email = ?",
                (email,),
            ).fetchone()
        return self.rowToUser(row)

    def getUserByUsername(self, username: str) -> Optional[UserRecord]:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        return self.rowToUser(row)

    def authenticateUser(self, username: str, password: str) -> Optional[UserRecord]:
        user = self.getUserByUsername(username)
        if user is None:
            return None

        if not self.verifyPassword(password, user.password_hash):
            return None

        with closing(self.connect()) as connection:
            connection.execute(
                """
                UPDATE users
                SET last_login_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (user.id,),
            )
            connection.commit()

        return self.getUserById(user.id)

    def listUsers(self) -> list[UserRecord]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM users ORDER BY created_at ASC, id ASC"
            ).fetchall()
        return [self.rowToUser(row) for row in rows if row is not None]

    def getAllUsers(self) -> list[UserRecord]:
        return self.listUsers()

    @staticmethod
    def rowToUser(row: sqlite3.Row | None) -> Optional[UserRecord]:
        if row is None:
            return None

        return UserRecord(
            id=row["id"],
            username=row["username"],
            email=row["email"],
            password_hash=row["password_hash"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_login_at=row["last_login_at"],
        )


def createUserDatabase(dbPath: Path | str = DEFAULT_DB_PATH) -> UserDatabase:
    return UserDatabase(dbPath=dbPath)


def createUser(
    username: str,
    email: str,
    password: str,
    dbPath: Path | str = DEFAULT_DB_PATH,
) -> UserRecord:
    database = createUserDatabase(dbPath=dbPath)
    return database.createUser(username=username, email=email, password=password)
