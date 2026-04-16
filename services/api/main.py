import uvicorn
from fastapi import FastAPI, HTTPException
from sqlmodel import SQLModel, Session, create_engine, select
import bcrypt


# ── Database ──────────────────────────────────
engine = create_engine("sqlite:///./users.db", connect_args={"check_same_thread": False})

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


@app.post("/data-patches", response_model=DataPatchResponse, status_code=201)
def create_data_patch(body: DataPatchCreate):
    with Session(engine) as session:
        data_patch = DataPatch(
            file_path=body.file_path,
            source_image=body.source_image,
            group=body.group,
        )
        session.add(data_patch)
        session.commit()
        session.refresh(data_patch)
        return data_patch


@app.get("/data-patches", response_model=list[DataPatchResponse])
def get_all_data_patches():
    with Session(engine) as session:
        return session.exec(select(DataPatch)).all()


@app.get("/data-patches/{data_patch_id}", response_model=DataPatchResponse)
def get_data_patch(data_patch_id: int):
    with Session(engine) as session:
        data_patch = session.get(DataPatch, data_patch_id)
        if not data_patch:
            raise HTTPException(404, "Data patch not found")
        return data_patch

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
