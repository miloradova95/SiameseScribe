import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]
sys.path.append(str(PROJECT_ROOT))

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import api
from services.ML.app.chroma_client import get_chroma_client, get_or_create_collection
from services.ML.app.services.SiameseNetwork import SiameseNetwork
from services.ML.app.services.segmentation.segmentation_service import SegmentationService

MODEL_PATH        = PROJECT_ROOT / "data" / "models" / "trainedModel.pth"
CHROMA_PATH       = PROJECT_ROOT / "data" / "chromaDB" / "data" / "chroma_store"
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "patches_v1")
EMBEDDING_DIM     = 128

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5175",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SiameseNetwork(embedding_dim=EMBEDDING_DIM).to(device)
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Siamese model loaded from {MODEL_PATH}")
    else:
        print(f"Warning: no checkpoint found at {MODEL_PATH}, using random weights")
    model.eval()
    app.state.model = model
    app.state.device = device

    try:
        seg_service = SegmentationService()
        app.state.seg_service = seg_service
    except Exception as e:
        print(f"Warning: segmentation model failed to load: {e}")
        app.state.seg_service = None

    try:
        client = get_chroma_client(str(CHROMA_PATH))
        collection = get_or_create_collection(client, CHROMA_COLLECTION)
        print(f"ChromaDB collection '{CHROMA_COLLECTION}' loaded ({collection.count()} embeddings)")
        app.state.collection = collection
    except Exception as e:
        print(f"Warning: ChromaDB collection '{CHROMA_COLLECTION}' not available: {e}")
        app.state.collection = None

    yield


app = FastAPI(title="ML Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api.router)


@app.get("/")
def root():
    return {"message": "ML service is running"}