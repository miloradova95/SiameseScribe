import sys
from contextlib import asynccontextmanager
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
sys.path.append(str(PROJECT_ROOT))

import torch
from fastapi import FastAPI

from app.routes import api
from services.ML.app.services.SiameseNetwork import SiameseNetwork

MODEL_PATH = PROJECT_ROOT / "data" / "models" / "trainedModel.pth"
EMBEDDING_DIM = 128


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SiameseNetwork(embedding_dim=EMBEDDING_DIM).to(device)

    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Warning: no model checkpoint found at {MODEL_PATH}, using random weights")

    model.eval()
    app.state.model = model
    app.state.device = device
    yield


app = FastAPI(title="ML Service", lifespan=lifespan)
app.include_router(api.router)


@app.get("/")
def root():
    return {"message": "ML service is running"}
