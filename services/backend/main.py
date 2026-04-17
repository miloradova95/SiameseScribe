from pathlib import Path
import sys
import uvicorn
from fastapi import FastAPI
from sqlmodel import SQLModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from services.backend.database import engine
from services.backend.routes.images import router as images_router
from services.backend.routes.patches import router as patches_router

app = FastAPI(title="SiameseScribe API")

@app.on_event("startup")
def startup():
    SQLModel.metadata.create_all(engine)

app.include_router(images_router)
app.include_router(patches_router)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)