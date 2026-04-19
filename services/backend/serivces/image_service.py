import random as _random
from sqlmodel import Session, select
from services.backend.sqlDB.images import Image


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