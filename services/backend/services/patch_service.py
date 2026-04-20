from sqlmodel import Session, select
from services.backend.sqlDB.patches import Patch


def get_all(session: Session) -> list[Patch]:
    return session.exec(select(Patch)).all()


def get_by_id(session: Session, patch_id: int) -> Patch | None:
    return session.get(Patch, patch_id)


def get_by_image_id(session: Session, image_id: int) -> list[Patch]:
    return session.exec(select(Patch).where(Patch.source_image_id == image_id)).all()

def get_by_file_name(session: Session, file_name: str) -> Patch | None:
    return session.exec(select(Patch).where(Patch.file_path.contains(file_name))).first()