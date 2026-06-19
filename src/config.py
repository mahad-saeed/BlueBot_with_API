"""
Shared configuration constants for BlueBot's local RAG pipeline.
"""

from __future__ import annotations

from pathlib import Path

# Project root is the parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ChromaDB persistent directory in project root (not inside src/)
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Collection where policy chunks + embeddings are stored
COLLECTION_NAME = "airblue_policies"

# Embedding model used for both indexing and retrieval
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
