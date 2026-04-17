from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlmodel import Session
from services.backend.serivces import image_service
from services.backend.sqlDB.images import Image
from services.backend.sqlDB.patches import Patch
from services.backend.serivces import patch_service
from .deps import get_session
from typing import Optional
import os
import shutil
import uuid

UPLOAD_DIR = "data/dataset/preprocessed/TEMP"

router = APIRouter(prefix="/images", tags=["images"])


@router.get("", response_model=list[Image])
def get_all_images(session: Session = Depends(get_session)):
    return image_service.get_all(session)


@router.post("/upload", response_model=Image)
def upload_image(
    file: UploadFile = File(...),
    group: Optional[str] = Form(default=None),
    session: Session = Depends(get_session),
):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    ext = os.path.splitext(file.filename)[1]
    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return image_service.create(session, file_name=file.filename, file_path=file_path, group=group)


@router.get("/{image_id}", response_model=Image)
def get_image_by_id(image_id: int, session: Session = Depends(get_session)):
    image = image_service.get_by_id(session, image_id)
    if not image:
        raise HTTPException(404, "Image not found")
    return image


@router.get("/{image_id}/file")
def get_image_file(image_id: int, session: Session = Depends(get_session)):
    image = image_service.get_by_id(session, image_id)
    if not image:
        raise HTTPException(404, "Image not found")
    if not os.path.isfile(image.filePath):
        raise HTTPException(404, "File not found on disk")
    return FileResponse(image.filePath)


@router.get("/{image_id}/patches", response_model=list[Patch])
def get_patches_by_image_id(image_id: int, session: Session = Depends(get_session)):
    if not image_service.get_by_id(session, image_id):
        raise HTTPException(404, "Image not found")
    return patch_service.get_by_image_id(session, image_id)