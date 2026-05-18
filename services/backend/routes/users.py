from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select, or_
from sqlalchemy import func

from services.backend.routes.deps import get_session, get_current_user, require_admin
from services.backend.schemas.auth import UserCreate, UserResponse
from services.backend.services.auth_service import hash_password
from services.backend.sqlDB.images import Image
from services.backend.sqlDB.users import User

router = APIRouter(prefix="/users", tags=["users"])


def _require_self_or_admin(user_id: int, current_user: User) -> None:
    if current_user.id != user_id and current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed to access this user's bookmarks")


@router.post("", response_model=UserResponse)
def create_user(
    user_data: UserCreate,
    session: Session = Depends(get_session),
    _: User = Depends(require_admin),
):
    normalized_username = user_data.username.lower()
    existing = session.exec(
        select(User).where(
            or_(
                func.lower(User.username) == normalized_username,
                User.email == user_data.email,
            )
        )
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


@router.get("/{user_id}/bookmarks", response_model=list[Image])
def get_user_bookmarks(
    user_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    _require_self_or_admin(user_id, current_user)

    target_user = session.get(User, user_id)
    if not target_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    bookmark_ids = [image_id for image_id in (target_user.bookmarks or []) if isinstance(image_id, int)]
    if not bookmark_ids:
        return []

    images = session.exec(select(Image).where(Image.id.in_(bookmark_ids))).all()
    images_by_id = {image.id: image for image in images}
    return [images_by_id[image_id] for image_id in bookmark_ids if image_id in images_by_id]


@router.post("/{user_id}/bookmarks/{image_id}", response_model=list[Image])
def add_user_bookmark(
    user_id: int,
    image_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    _require_self_or_admin(user_id, current_user)

    target_user = session.get(User, user_id)
    if not target_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    image = session.get(Image, image_id)
    if not image:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    bookmarks = [bookmark_id for bookmark_id in (target_user.bookmarks or []) if isinstance(bookmark_id, int)]
    if image_id not in bookmarks:
        target_user.bookmarks = [*bookmarks, image_id]
        session.add(target_user)
        session.commit()
        session.refresh(target_user)

    return get_user_bookmarks(user_id=user_id, session=session, current_user=current_user)


@router.delete("/{user_id}/bookmarks/{image_id}", response_model=list[Image])
def remove_user_bookmark(
    user_id: int,
    image_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    _require_self_or_admin(user_id, current_user)

    target_user = session.get(User, user_id)
    if not target_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    bookmarks = [bookmark_id for bookmark_id in (target_user.bookmarks or []) if isinstance(bookmark_id, int)]
    next_bookmarks = [bookmark_id for bookmark_id in bookmarks if bookmark_id != image_id]

    if next_bookmarks != bookmarks:
        target_user.bookmarks = next_bookmarks
        session.add(target_user)
        session.commit()
        session.refresh(target_user)

    return get_user_bookmarks(user_id=user_id, session=session, current_user=current_user)


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

@router.patch("/{user_id}/activate", response_model=UserResponse)
def activate_user(
    user_id: int,
    session: Session = Depends(get_session),
    _: User = Depends(require_admin),
):
    user = session.get(User, user_id)

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    user.is_active = True
    session.add(user)
    session.commit()
    session.refresh(user)

    return user

@router.delete("/{user_id}")
def delete_user(
    user_id: int,
    session: Session = Depends(get_session),
    current_admin: User = Depends(require_admin),
):
    user = session.get(User, user_id)

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if user.id == current_admin.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete yourself")

    session.delete(user)
    session.commit()

    return {"detail": "User deleted"}
