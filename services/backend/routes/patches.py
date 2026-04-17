from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session
from services.backend.sqlDB.patches import Patch
from services.backend.serivces import patch_service
from .deps import get_session

router = APIRouter(prefix="/patches", tags=["patches"])


@router.get("", response_model=list[Patch])
def get_all_patches(session: Session = Depends(get_session)):
    return patch_service.get_all(session)


@router.get("/{patch_id}", response_model=Patch)
def get_patch_by_id(patch_id: int, session: Session = Depends(get_session)):
    patch = patch_service.get_by_id(session, patch_id)
    if not patch:
        raise HTTPException(404, "Patch not found")
    return patch