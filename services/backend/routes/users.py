from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select, or_

from services.backend.routes.deps import get_session, require_admin
from services.backend.schemas.auth import UserCreate, UserResponse
from services.backend.services.auth_service import hash_password
from services.backend.sqlDB.users import User

router = APIRouter(prefix="/users", tags=["users"])


@router.post("", response_model=UserResponse)
def create_user(
    user_data: UserCreate,
    session: Session = Depends(get_session),
    _: User = Depends(require_admin),
):
    existing = session.exec(
        select(User).where(or_(User.username == user_data.username, User.email == user_data.email))
    ).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username or email already exists")
    user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
        role=user_data.role,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


@router.get("", response_model=list[UserResponse])
def list_users(
    session: Session = Depends(get_session),
    _: User = Depends(require_admin),
):
    return session.exec(select(User)).all()


@router.patch("/{user_id}/deactivate", response_model=UserResponse)
def deactivate_user(
    user_id: int,
    session: Session = Depends(get_session),
    current_admin: User = Depends(require_admin),
):
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if user.id == current_admin.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot deactivate yourself")
    user.is_active = False
    session.add(user)
    session.commit()
    session.refresh(user)
    return user
