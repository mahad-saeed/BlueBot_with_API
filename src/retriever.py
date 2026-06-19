"""
Retrieval pipeline for BlueBot RAG.

This module queries ChromaDB using the same embedding model used at index time.
It also exposes an `is_relevant` guardrail signal so downstream code can avoid
calling the LLM when policy context is not a reliable match.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import chromadb
from sentence_transformers import SentenceTransformer


# Ensure project root is importable when running: python src/retriever.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME  # noqa: E402


# Calibrated starter threshold from observed distance separation in this corpus:
# on-topic policy queries roughly fell in ~0.75-1.38 while clear off-topic were
# ~1.79+, so we keep this value inside that gap and keep re-validating with the
# calibration routine below as documents evolve.
DISTANCE_THRESHOLD = 1.58

# Module-level singletons: loaded once per process for lower latency.
_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
_CLIENT = chromadb.PersistentClient(path=str(CHROMA_DIR))
_COLLECTION = _CLIENT.get_or_create_collection(name=COLLECTION_NAME)


class EmptyCollectionError(RuntimeError):
    """Raised when retrieval is attempted before embeddings are indexed."""


@dataclass(frozen=True)
class RetrievedChunk:
    """Single retrieved policy chunk with metadata and raw distance."""

    text: str
    source: str
    chunk_id: str
    distance: float


@dataclass(frozen=True)
class RetrievalResult:
    """
    Retrieval output used by downstream generation control flow.

    - chunks: top-k results as returned by Chroma (distance included)
    - is_relevant: True only when best match passes DISTANCE_THRESHOLD
    """

    chunks: list[RetrievedChunk]
    is_relevant: bool


def _ensure_collection_ready() -> None:
    """
    Ensure collection contains indexed documents before querying.

    Raises:
        EmptyCollectionError: if no docs are available in the collection.
    """
    if _COLLECTION.count() == 0:
        raise EmptyCollectionError(
            "Chroma collection is empty. Run src/embedder.py first to index policy chunks."
        )


def retrieve(query: str, k: int = 3) -> RetrievalResult:
    """
    Retrieve top-k policy chunks for a query using manual query embeddings.

    This collection was created without a Chroma embedding_function, and documents
    were embedded manually with SentenceTransformer(all-MiniLM-L6-v2). Therefore we
    must also embed queries manually and pass `query_embeddings` to `collection.query`.

    Distance space note:
    - Because the collection did not specify `hnsw:space`, Chroma uses its default
      distance space (currently L2). Lower distance means more similar.

    Args:
        query: User question text.
        k: Number of chunks to retrieve (default 3).

    Returns:
        RetrievalResult with chunks and relevance flag.
    """
    _ensure_collection_ready()

    cleaned_query = query.strip()
    if not cleaned_query:
        return RetrievalResult(chunks=[], is_relevant=False)

    safe_k = max(1, k)

    query_embedding = _MODEL.encode(cleaned_query).tolist()
    raw = _COLLECTION.query(
        query_embeddings=[query_embedding],
        n_results=safe_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = raw.get("documents", [[]])[0] or []
    metadatas = raw.get("metadatas", [[]])[0] or []
    distances = raw.get("distances", [[]])[0] or []

    chunks: list[RetrievedChunk] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        chunk_id = ""
        source = ""
        if isinstance(metadata, dict):
            chunk_id = str(metadata.get("chunk_id", ""))
            source = str(metadata.get("source", ""))

        chunks.append(
            RetrievedChunk(
                text=str(document),
                source=source,
                chunk_id=chunk_id,
                distance=float(distance),
            )
        )

    if not chunks:
        return RetrievalResult(chunks=[], is_relevant=False)

    best_distance = min(chunk.distance for chunk in chunks)
    is_relevant = best_distance <= DISTANCE_THRESHOLD
    return RetrievalResult(chunks=chunks, is_relevant=is_relevant)


def _print_calibration_queries() -> None:
    """
    Print top-1 distances for verified calibration prompts.

    Use this output to tune DISTANCE_THRESHOLD based on observed separation.
    """
    on_topic_queries = [
        "What is the baggage allowance for international flights?",
        "How do I get a refund?",
        "Can I change my flight after booking?",
        "What is the baggage allowance for Flexi fare?",
    ]
    not_well_covered_queries = [
        "What documents do I need to check in?",
    ]
    airline_adjacent_uncovered_queries = [
        "Does Airblue fly to Tokyo?",
        "What aircraft does Airblue use on the Karachi route?",
        "Is Airblue a Star Alliance member?",
    ]
    off_topic_queries = [
        "What is the capital of France?",
        "Write me a poem about the ocean",
    ]

    print("\nCalibration helper: inspect top-1 distances before setting threshold.")
    print("Set DISTANCE_THRESHOLD from these numbers, not by guesswork.\n")

    print("Category: verified answerable from current docs")
    for q in on_topic_queries:
        result = retrieve(q, k=1)
        top = result.chunks[0].distance if result.chunks else float("inf")
        print(
            "[EXPECT TRUE] "
            f"{q}\n    top1_distance={top:.6f} is_relevant={result.is_relevant}"
        )

    print("\nCategory: airline-related but not covered in current docs")
    for q in not_well_covered_queries:
        result = retrieve(q, k=1)
        top = result.chunks[0].distance if result.chunks else float("inf")
        print(
            "[EXPECT FALSE] "
            f"{q}\n    top1_distance={top:.6f} is_relevant={result.is_relevant}"
        )

    print("\nCategory: airline-adjacent but uncovered")
    for q in airline_adjacent_uncovered_queries:
        result = retrieve(q, k=1)
        top = result.chunks[0].distance if result.chunks else float("inf")
        print(
            "[EXPECT FALSE] "
            f"{q}\n    top1_distance={top:.6f} is_relevant={result.is_relevant}"
        )

    print("\nCategory: clearly off-topic")
    for q in off_topic_queries:
        result = retrieve(q, k=1)
        top = result.chunks[0].distance if result.chunks else float("inf")
        print(
            "[EXPECT FALSE] "
            f"{q}\n    top1_distance={top:.6f} is_relevant={result.is_relevant}"
        )


def _print_query_result(query: str, k: int = 3) -> None:
    """Print retrieval output for a single ad-hoc query."""
    print(f"\nQuery: {query}")
    retrieval = retrieve(query, k=k)
    print(f"is_relevant: {retrieval.is_relevant}")
    for idx, chunk in enumerate(retrieval.chunks, start=1):
        preview = chunk.text[:180] + ("..." if len(chunk.text) > 180 else "")
        print(
            f"  {idx}. chunk_id={chunk.chunk_id} source={chunk.source} "
            f"distance={chunk.distance:.6f}"
        )
        print(f"     text={preview}")


if __name__ == "__main__":
    # CLI mode: python src/retriever.py "your question here"
    # If a query argument is provided, run only that query and exit.
    if len(sys.argv) > 1:
        cli_query = sys.argv[1]
        _print_query_result(cli_query, k=3)
    else:
        _print_calibration_queries()
