from pydantic import BaseModel


class AdminImageListItem(BaseModel):
    id: int
    fileName: str
    filePath: str
    group: str | None = None
    userId: int | None = None
    username: str | None = None
    user_email: str | None = None
    patches: list[int] | None = None
    toBeDeleted: int = 0
