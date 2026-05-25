import asyncio
import os
from pathlib import Path
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from services.backend.routes.deps import require_admin
from services.backend.sqlDB.users import User

router = APIRouter(prefix="/admin/dataset", tags=["dataset-sync"])

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = _PROJECT_ROOT / "data"

ALLOWED_PREFIXES = [
    "patches/train/",
    "patches/test/",
    "dataset/preprocessed/train/",
    "dataset/preprocessed/test/",
]

_TRACKED_DIRS = [
    "patches/train",
    "patches/test",
    "dataset/preprocessed/train",
    "dataset/preprocessed/test",
    "models",
]


@router.get("/info")
def dataset_info(count_files: bool = False, _: User = Depends(require_admin)):
    """High-level snapshot of what exists in the server's data directory.

    Pass ?count_files=true to include file counts (slow on large directories).
    """
    dirs = {}
    for rel in _TRACKED_DIRS:
        d = DATA_ROOT / rel
        if d.exists():
            count = sum(1 for f in d.rglob("*") if f.is_file()) if count_files else None
            dirs[rel] = {"exists": True, "file_count": count}
        else:
            dirs[rel] = {"exists": False, "file_count": None}
    return {"data_root": str(DATA_ROOT), "directories": dirs}


def _build_inventory() -> dict[str, int]:
    files: dict[str, int] = {}
    for prefix in ALLOWED_PREFIXES:
        d = DATA_ROOT / prefix.rstrip("/")
        if not d.exists():
            continue
        for f in d.rglob("*"):
            if f.is_file():
                files[f.relative_to(DATA_ROOT).as_posix()] = f.stat().st_size
    return files


@router.get("/inventory")
async def dataset_inventory(_: User = Depends(require_admin)):
    """Flat map of {relative_path: size_bytes} for all files in upload directories."""
    files = await asyncio.to_thread(_build_inventory)
    return {"files": files, "total": len(files)}


@router.post("/upload-file", status_code=201)
async def upload_file(
    relative_path: str = Form(...),
    file: UploadFile = File(...),
    _: User = Depends(require_admin),
):
    """Upload a single file to a whitelisted path under data/."""
    if ".." in relative_path or relative_path.startswith("/"):
        raise HTTPException(400, "Invalid path")

    if not any(relative_path.startswith(prefix) for prefix in ALLOWED_PREFIXES):
        raise HTTPException(400, f"Path must start with one of: {ALLOWED_PREFIXES}")

    dest = (DATA_ROOT / relative_path).resolve()

    # Belt-and-suspenders: resolved path must stay under DATA_ROOT
    try:
        dest.relative_to(DATA_ROOT.resolve())
    except ValueError:
        raise HTTPException(400, "Path resolves outside data root")

    dest.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: stream to .tmp, then rename
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with open(tmp, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                out.write(chunk)
        os.replace(tmp, dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    return {"path": relative_path, "size": dest.stat().st_size}
