from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlmodel import Session
import os
from services.backend.sqlDB.patches import Patch
from services.backend.services import patch_service
from .deps import get_session

router = APIRouter(prefix="/patches", tags=["patches"])


@router.get("", response_model=list[Patch])
def get_all_patches(session: Session = Depends(get_session)):
    return patch_service.get_all(session)


@router.get("/by-image/{image_id}", response_model=list[Patch])
def get_patches_by_image_id(image_id: int, session: Session = Depends(get_session)):
    return patch_service.get_by_image_id(session, image_id)


@router.get("/{patch_id}", response_model=Patch)
def get_patch_by_id(patch_id: int, session: Session = Depends(get_session)):
    patch = patch_service.get_by_id(session, patch_id)
    if not patch:
        raise HTTPException(404, "Patch not found")
    return patch


@router.get("/{patch_id}/file")
def get_patch_file(patch_id: int, session: Session = Depends(get_session)):
    patch = patch_service.get_by_id(session, patch_id)
    if not patch:
        raise HTTPException(404, "Patch not found")
    if not os.path.isfile(patch.file_path):
        raise HTTPException(404, "File not found on disk")
    return FileResponse(patch.file_path)