import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr
from sqlmodel import SQLModel, Field, Session, create_engine, select
import bcrypt
from typing import Optional
from datetime import datetime, timezone

# ── Database ──────────────────────────────────
engine = create_engine("sqlite:///./users.db", connect_args={"check_same_thread": False})

# ── User table ────────────────────────────────
class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(unique=True, index=True)
    email: str = Field(unique=True, index=True)
    hashed_password: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# ── Schemas ───────────────────────────────────
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime

# ── App ───────────────────────────────────────
app = FastAPI(title="SiameseScribe API")

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

@app.on_event("startup")
def startup():
    SQLModel.metadata.create_all(engine)

@app.post("/users", response_model=UserResponse, status_code=201)
def create_user(body: UserCreate):
    with Session(engine) as session:
        if session.exec(select(User).where(User.username == body.username)).first():
            raise HTTPException(409, "Username already taken")
        if session.exec(select(User).where(User.email == body.email)).first():
            raise HTTPException(409, "Email already registered")
        user = User(
            username=body.username,
            email=body.email,
            hashed_password=hash_password(body.password),
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user

@app.get("/users", response_model=list[UserResponse])
def get_all_users():
    with Session(engine) as session:
        return session.exec(select(User)).all()

@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int):
    with Session(engine) as session:
        user = session.get(User, user_id)
        if not user:
            raise HTTPException(404, "User not found")
        return user

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
