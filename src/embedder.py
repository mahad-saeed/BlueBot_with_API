"""
Embedding pipeline for BlueBot RAG.

This script:
1) Builds chunks from src.chunker.create_chunks()
2) Generates embeddings with sentence-transformers/all-MiniLM-L6-v2
3) Persists documents + embeddings + metadata in ChromaDB
"""

from __future__ import annotations

from pathlib import Path
import sys

import chromadb
from sentence_transformers import SentenceTransformer


# Ensure project root is importable when running: python src/embedder.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunker import create_chunks  # noqa: E402
from src.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME  # noqa: E402


def embed_and_store() -> tuple[int, str]:
    """
    Create embeddings for all chunks and store them in a persistent Chroma collection.

    Returns:
        tuple[int, str]:
            - number of chunks embedded in this run
            - status message
    """
    # Build chunks from cleaned policy text files.
    chunks = create_chunks()
    if not chunks:
        return 0, "No chunks found to embed."

    # Use a persistent Chroma client rooted at project_root/chroma_db.
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Skip work if this collection already contains documents.
    if collection.count() > 0:
        print("Collection already exists, skipping")
        return 0, "Collection already exists, skipping"

    # Extract payloads used for collection.add().
    ids = [chunk["chunk_id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [{"source": chunk["source"], "chunk_id": chunk["chunk_id"]} for chunk in chunks]

    # Generate embeddings manually, then pass them into ChromaDB.
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(documents, show_progress_bar=True).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return len(chunks), "ChromaDB collection created"


if __name__ == "__main__":
    embedded_count, status = embed_and_store()

    print(f"How many chunks were embedded: {embedded_count}")
    print(f"Confirmation that ChromaDB collection was created: {status}")
