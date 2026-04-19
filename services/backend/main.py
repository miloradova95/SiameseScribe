from pathlib import Path
import csv
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, Session, select

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from services.backend.database import engine
from services.backend.routes.images import router as images_router
from services.backend.routes.patches import router as patches_router
from services.backend.sqlDB.images import Image
from services.backend.sqlDB.patches import Patch

PREPROCESSED_DIR = PROJECT_ROOT / "data" / "dataset" / "preprocessed"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

app = FastAPI(title="SiameseScribe API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PATCHES_DIR = PROJECT_ROOT / "data" / "patches"

@app.on_event("startup")
def startup():
    SQLModel.metadata.create_all(engine)
    _seed_images()
    _seed_patches()

def _seed_images():
    with Session(engine) as session:
        already_seeded = session.exec(select(Image)).first()
        if already_seeded:
            return

        images = []
        for split in ("train", "test"):
            split_dir = PREPROCESSED_DIR / split
            if not split_dir.exists():
                continue
            for group_dir in sorted(split_dir.iterdir()):
                if not group_dir.is_dir():
                    continue
                group = group_dir.name
                for img_path in sorted(group_dir.iterdir()):
                    if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                        images.append(Image(
                            fileName=img_path.name,
                            filePath=str(img_path.resolve()),
                            group=group,
                        ))

        session.add_all(images)
        session.commit()

def _seed_patches():
    with Session(engine) as session:
        already_seeded = session.exec(select(Patch)).first()
        if already_seeded:
            return

        patches = []
        for mode in ("train", "test"):
            csv_path = PATCHES_DIR / f"patches_{mode}_metadata.csv"
            if not csv_path.exists():
                continue
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    source_image_id = row["source_image_id"]
                    if not source_image_id:
                        continue
                    patches.append(Patch(
                        source_image_id=int(source_image_id),
                        file_path=str((PATCHES_DIR / mode / row["patch_filename"]).resolve()),
                        bbox={"x": int(row["x"]), "y": int(row["y"]), "width": 128, "height": 128},
                        group=row["group"] or None,
                        codex=row["codex"] or None,
                        pen_flourishing_percent=float(row["pen_flourishing_percent"]) if row["pen_flourishing_percent"] not in ("", "None") else None,
                    ))

        session.add_all(patches)
        session.commit()


app.include_router(images_router)
app.include_router(patches_router)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)