from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from sqlalchemy import func

from services.backend.routes.deps import get_session, get_current_user
from services.backend.schemas.auth import LoginRequest, TokenResponse, UserResponse
from services.backend.services.auth_service import verify_password, create_access_token
from services.backend.sqlDB.users import User

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest, session: Session = Depends(get_session)):
    normalized_username = request.username.lower()
    user = session.exec(
        select(User).where(func.lower(User.username) == normalized_username)
    ).first()
    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    if not user.is_active or user.toBeDeleted:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is inactive")
    token = create_access_token({"sub": str(user.id)})
    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user
