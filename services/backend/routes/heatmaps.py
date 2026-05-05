from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix="/heatmaps", tags=["heatmaps"])

_HEATMAPS_DIR = Path(__file__).resolve().parents[3] / "data" / "heatmaps"


@router.get("/{filename}")
def get_heatmap(filename: str):
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = _HEATMAPS_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Heatmap not found")
    return FileResponse(str(path), media_type="image/png")
