import random as _random
import os
import shutil
import uuid
from pathlib import Path
from fastapi import UploadFile
from sqlmodel import Session, select
from services.backend.sqlDB.images import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPLOAD_DIR = PROJECT_ROOT / "data" / "dataset" / "preprocessed" / "TEMP"


def get_all(session: Session) -> list[Image]:
    return session.exec(select(Image)).all()


def get_by_id(session: Session, image_id: int) -> Image | None:
    return session.get(Image, image_id)


def get_id_by_filename(session: Session, file_name: str) -> int | None:
    result = session.exec(select(Image).where(Image.fileName == file_name)).first()
    return result.id if result else None


def get_random(session: Session) -> Image | None:
    all_ids = session.exec(select(Image.id)).all()
    if not all_ids:
        return None
    return session.get(Image, _random.choice(all_ids))


def create(session: Session, file_name: str, file_path: str, group: str | None = None) -> Image:
    image = Image(fileName=file_name, filePath=file_path, group=group)
    session.add(image)
    session.commit()
    session.refresh(image)
    return image


def save_upload(session: Session, file: UploadFile, group: str | None = None) -> Image:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    extension = os.path.splitext(file.filename or "")[1]
    unique_name = f"{uuid.uuid4().hex}{extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)

    with open(file_path, "wb") as destination:
        shutil.copyfileobj(file.file, destination)

    return create(session, file_name=file.filename, file_path=file_path, group=group)


def resolve_file_path(image: Image) -> Path:
    return PROJECT_ROOT.parent / image.filePath.replace("\\", "/")
