import chromadb
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CHROMA_PATH = str(_PROJECT_ROOT / "data" / "chromaDB" / "chromaStore")

def getChromaClient(persist_path: str = _DEFAULT_CHROMA_PATH) -> chromadb.ClientAPI:

    client = chromadb.PersistentClient(path=persist_path) #local db stored in path
    return client


def getOrCreateCollection(client: chromadb.ClientAPI, name: str = "paintings"): #collection = table
    
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
        # metadata: hnsw: similarity metric defined (ANN (approx. nearest neighbor) and cosine similarity)
    )
    return collection
