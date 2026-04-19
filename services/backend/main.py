from pathlib import Path
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

PREPROCESSED_DIR = PROJECT_ROOT / "data" / "dataset" / "preprocessed"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

app = FastAPI(title="SiameseScribe API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    SQLModel.metadata.create_all(engine)
    _seed_images()

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

app.include_router(images_router)
app.include_router(patches_router)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)