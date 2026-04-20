import chromadb
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CHROMA_PATH = str(_PROJECT_ROOT / "data" / "chromaDB" / "data" / "chroma_store")


def get_chroma_client(persist_path: str = _DEFAULT_CHROMA_PATH) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=persist_path)


def get_or_create_collection(client: chromadb.ClientAPI, name: str = "patches_v1"):
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
