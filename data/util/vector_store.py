import sys
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from agents.shared.embedding_models import EMBEDDING_MODELS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path="data/chroma")


def get_or_create_collection(ticker: str) -> chromadb.Collection:
    client = get_chroma_client()

    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODELS["hf_embed_fast"],
        device="cpu",
        normalize_embeddings=True,
    )

    return client.get_or_create_collection(
        name=f"filings_{ticker.lower()}",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def collection_exists(ticker: str) -> bool:
    client = get_chroma_client()
    collection_name = f"filings_{ticker.lower()}"

    try:
        collections = client.list_collections()
        return any(c.name == collection_name for c in collections)
    except Exception:
        return False


def get_collection_stats(ticker: str) -> dict:
    if not collection_exists(ticker):
        return {"exists": False, "document_count": 0}

    collection = get_or_create_collection(ticker)
    count = collection.count()

    results = collection.get(limit=1, include=["metadatas"])
    latest_date = None

    if results["metadatas"]:
        latest_date = results["metadatas"][0].get("filing_date")

    return {"exists": True, "document_count": count, "latest_filing_date": latest_date}


def delete_collection(ticker: str) -> bool:
    client = get_chroma_client()
    collection_name = f"filings_{ticker.lower()}"

    try:
        client.delete_collection(collection_name)
        return True
    except Exception:
        return False
