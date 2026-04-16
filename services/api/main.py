from pathlib import Path
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from sqlmodel import SQLModel, Session, create_engine, select

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from services.api.sqlDB.images import Image
from services.api.sqlDB.patches import Patch


# ── Database ──────────────────────────────────
DB_PATH = Path(__file__).resolve().parent / "images.db"
engine = create_engine(f"sqlite:///{DB_PATH.as_posix()}", connect_args={"check_same_thread": False})

# ── App ───────────────────────────────────────
app = FastAPI(title="SiameseScribe API")

@app.on_event("startup")
def startup():
    SQLModel.metadata.create_all(engine)


@app.get("/images", response_model=list[Image])
def get_all_images():
    with Session(engine) as session:
        return session.exec(select(Image)).all()


@app.get("/images/{image_id}", response_model=Image)
def get_image_by_id(image_id: int):
    with Session(engine) as session:
        image = session.get(Image, image_id)
        if not image:
            raise HTTPException(404, "Image not found")
        return image


@app.get("/patches", response_model=list[Patch])
def get_all_patches():
    with Session(engine) as session:
        return session.exec(select(Patch)).all()


@app.get("/patches/{patch_id}", response_model=Patch)
def get_patch_by_id(patch_id: int):
    with Session(engine) as session:
        patch = session.get(Patch, patch_id)
        if not patch:
            raise HTTPException(404, "Patch not found")
        return patch


@app.get("/images/{image_id}/patches", response_model=list[Patch])
def get_patches_by_image_id(image_id: int):
    with Session(engine) as session:
        image = session.get(Image, image_id)
        if not image:
            raise HTTPException(404, "Image not found")
        return session.exec(select(Patch).where(Patch.image_id == image_id)).all()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
