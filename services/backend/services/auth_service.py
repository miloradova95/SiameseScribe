import os
import bcrypt
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from fastapi import HTTPException, status

ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 8


def _secret_key() -> str:
    return os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(data: dict) -> str:
    payload = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HOURS)
    payload["exp"] = expire
    return jwt.encode(payload, _secret_key(), algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, _secret_key(), algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
