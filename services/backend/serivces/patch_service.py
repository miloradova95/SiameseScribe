from sqlmodel import Session, select
from services.backend.sqlDB.patches import Patch


def get_all(session: Session) -> list[Patch]:
    return session.exec(select(Patch)).all()


def get_by_id(session: Session, patch_id: int) -> Patch | None:
    return session.get(Patch, patch_id)


def get_by_image_id(session: Session, image_id: int) -> list[Patch]:
    return session.exec(select(Patch).where(Patch.image_id == image_id)).all()