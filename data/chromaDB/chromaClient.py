import chromadb

def get_chroma_client(persist_path: str = "./data/chromaDB/chromaStore") -> chromadb.ClientAPI:

    client = chromadb.PersistentClient(path=persist_path) #local db stored in path
    return client


def get_or_create_collection(client: chromadb.ClientAPI, name: str = "paintings"): #collection = table
    
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
        # metadata: hnsw: similarity metric defined (ANN (approx. nearest neighbor) and cosine similarity)
    )
    return collection
