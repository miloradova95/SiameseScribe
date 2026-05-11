from contextlib import asynccontextmanager
from pathlib import Path
import asyncio
import csv
import logging
import os
import sys
import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, Session, select
from sqlalchemy import text

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

from services.backend.database import engine
from services.backend.routes.images import router as images_router
from services.backend.routes.patches import router as patches_router
from services.backend.routes.feedback import router as feedback_router
from services.backend.routes.finetune import router as finetune_router
from services.backend.routes.auth import router as auth_router
from services.backend.routes.users import router as users_router
from services.backend.routes.heatmaps import router as heatmaps_router
from services.backend.routes.deps import get_current_user
from services.backend.sqlDB.images import Image
from services.backend.sqlDB.patches import Patch
from services.backend.sqlDB.feedback import Feedback
from services.backend.sqlDB.finetune_run import FinetuneRun  # registers table with SQLModel metadata
from services.backend.sqlDB.users import User
from services.backend.services import finetune_job
from services.backend.services.auth_service import hash_password

PREPROCESSED_DIR = PROJECT_ROOT / "data" / "dataset" / "preprocessed"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ──────────────────────────────
    SQLModel.metadata.create_all(engine)
    _migrate_images_user_id()
    _seed_images()
    _seed_patches()
    _seed_admin()
    _backfill_known_uploaded_image()
    scheduler_task = asyncio.create_task(_finetune_scheduler_loop())

    yield

    # ── shutdown ─────────────────────────────
    scheduler_task.cancel()
    try:
        await scheduler_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="SiameseScribe API", lifespan=lifespan)

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5175",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PATCHES_DIR = PROJECT_ROOT / "data" / "patches"

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
                        relative_path = img_path.resolve().relative_to(PROJECT_ROOT.parent)
                        images.append(Image(
                            fileName=img_path.name,
                            filePath=str(relative_path),
                            group=group,
                        ))
        session.add_all(images)
        session.commit()


def _migrate_images_user_id():
    with engine.begin() as connection:
        columns = connection.execute(text("PRAGMA table_info(images)")).fetchall()
        column_names = {column[1] for column in columns}

        if "userId" not in column_names:
            connection.execute(text('ALTER TABLE images ADD COLUMN "userId" INTEGER'))

        connection.execute(text('CREATE INDEX IF NOT EXISTS "ix_images_userId" ON images ("userId")'))


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
                    full_path = (PATCHES_DIR / mode / row["patch_filename"]).resolve()
                    relative_path = full_path.relative_to(PROJECT_ROOT.parent)
                    patches.append(Patch(
                        source_image_id=int(source_image_id),
                        file_path=str(relative_path),
                        bbox={"x": int(row["x"]), "y": int(row["y"]), "width": 128, "height": 128},
                        group=row["group"] or None,
                        codex=row["codex"] or None,
                        pen_flourishing_percent=float(row["pen_flourishing_percent"]) if row["pen_flourishing_percent"] not in ("", "None") else None,
                    ))
        session.add_all(patches)
        session.commit()


def _seed_admin():
    with Session(engine) as session:
        existing_admin = session.exec(select(User).where(User.role == "admin")).first()
        if existing_admin:
            return
        username = os.getenv("ADMIN_USERNAME", "admin")
        email = os.getenv("ADMIN_EMAIL", "admin@siamesescribe.local")
        password = os.getenv("ADMIN_PASSWORD", "changeme123")
        admin = User(
            username=username,
            email=email,
            hashed_password=hash_password(password),
            role="admin",
        )
        session.add(admin)
        session.commit()
        print(f"[startup] Admin user '{username}' created.")


def _backfill_known_uploaded_image():
    with engine.begin() as connection:
        connection.execute(
            text('UPDATE images SET "userId" = 1 WHERE id = 789 AND "userId" IS NULL')
        )


app.include_router(auth_router)
app.include_router(users_router, dependencies=[Depends(get_current_user)])
app.include_router(images_router)
app.include_router(patches_router)
app.include_router(feedback_router)
app.include_router(finetune_router)
app.include_router(heatmaps_router)


# ─────────────────────────────────────────────
# Finetune background scheduler
# ─────────────────────────────────────────────

def _finetune_scheduler_tick() -> None:
    """Single synchronous scheduler tick — runs in a thread pool via asyncio.to_thread."""
    with Session(engine) as session:
        run_id = finetune_job.evaluate_and_trigger(session, trigger_source="auto")
    if run_id is not None:
        finetune_job.run_automated_finetune_job(run_id)


async def _finetune_scheduler_loop() -> None:
    interval = int(os.getenv("FINETUNE_INTERVAL_MINUTES", "15")) * 60
    logger.info("Finetune scheduler started (interval=%ds)", interval)
    while True:
        await asyncio.sleep(interval)
        try:
            await asyncio.to_thread(_finetune_scheduler_tick)
        except Exception:
            logger.exception("Finetune scheduler tick failed")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
