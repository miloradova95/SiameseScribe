from sqlmodel import Session, select
from services.backend.sqlDB.images import Image


def get_all(session: Session) -> list[Image]:
    return session.exec(select(Image)).all()


def get_by_id(session: Session, image_id: int) -> Image | None:
    return session.get(Image, image_id)


def create(session: Session, file_name: str, file_path: str, group: str | None = None) -> Image:
    image = Image(fileName=file_name, filePath=file_path, group=group)
    session.add(image)
    session.commit()
    session.refresh(image)
    return image