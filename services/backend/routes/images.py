from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlmodel import Session
from services.backend.services import image_service
from services.backend.sqlDB.images import Image
from services.backend.sqlDB.patches import Patch
from services.backend.services import patch_service
from .deps import get_session, get_current_user
from services.backend.sqlDB.users import User
from typing import Optional

router = APIRouter(prefix="/images", tags=["images"])


@router.get("", response_model=list[Image])
def get_all_images(session: Session = Depends(get_session), _: User = Depends(get_current_user)):
    return image_service.get_all(session)


@router.post("/upload", response_model=Image)
def upload_image(
    file: UploadFile = File(...),
    group: Optional[str] = Form(default=None),
    session: Session = Depends(get_session),
    _: User = Depends(get_current_user),
):
    return image_service.save_upload(session, file=file, group=group)


@router.get("/random", response_model=Image)
def get_random_image(session: Session = Depends(get_session), _: User = Depends(get_current_user)):
    image = image_service.get_random(session)
    if not image:
        raise HTTPException(404, "No images found")
    return image


@router.get("/{image_id}", response_model=Image)
def get_image_by_id(image_id: int, session: Session = Depends(get_session), _: User = Depends(get_current_user)):
    image = image_service.get_by_id(session, image_id)
    if not image:
        raise HTTPException(404, "Image not found")
    return image


@router.get("/{image_id}/file")
def get_image_file(image_id: int, session: Session = Depends(get_session)):
    image = image_service.get_by_id(session, image_id)
    if not image:
        raise HTTPException(404, "Image not found")
    full_path = image_service.resolve_file_path(image)
    if not full_path.is_file():
        raise HTTPException(404, f"File not found on disk: {full_path}")
    return FileResponse(str(full_path))


@router.get("/{image_id}/patches", response_model=list[Patch])
def get_patches_by_image_id(image_id: int, session: Session = Depends(get_session), _: User = Depends(get_current_user)):
    if not image_service.get_by_id(session, image_id):
        raise HTTPException(404, "Image not found")
    return patch_service.get_by_image_id(session, image_id)
