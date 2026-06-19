"""
End-to-end BlueBot question pipeline.

Flow:
1) Retrieve policy chunks from local ChromaDB
2) If not relevant, return fallback without calling the LLM
3) If relevant, build constrained prompt and call the LLM (Groq by default,
   Ollama available as a documented local/production fallback)
"""

from __future__ import annotations

import os
import os
API_URL = os.getenv("API_URL", "http://localhost:8000/chat")
from pathlib import Path
import sys
import time

import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# Ensure project root is importable when running: python src/pipeline.py "question"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env from the project root regardless of current working directory.
load_dotenv(PROJECT_ROOT / ".env")

from src.config import EMBEDDING_MODEL_NAME  # noqa: E402

# Load embedding model once at import time; shared with retriever below.
_EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)

import src.retriever as _retriever  # noqa: E402

# Point retriever at this shared instance so retrieve() does not load a second copy.
_retriever._MODEL = _EMBEDDING_MODEL
retrieve = _retriever.retrieve


FALLBACK_MESSAGE = (
    "I don't have information about that. For assistance please contact Airblue "
    "support at 111-247-258 or visit airblue.com"
)

SYSTEM_PROMPT = SYSTEM_PROMPT = """You are BlueBot, a customer service assistant for Airblue Pakistan.
Answer using ONLY the exact facts in the CONTEXT below.
For baggage questions, state ONLY the specific fare type asked about.
Do not list or summarize other fare types.
Do not mention meals, seat selection, or BlueMiles unless the customer asks.
If the context does not contain the answer, say: "I don't have that information. Please contact Airblue support at 111-247-258."
Be concise. One or two sentences maximum."""

# --- LLM backend selection -------------------------------------------------
# "groq" (default, current path) or "ollama" (documented local/production path).
LLM_BACKEND = os.getenv("LLM_BACKEND", "groq").lower()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Either "llama-3.1-8b-instant" or "llama3-8b-8192" per the handoff notes.
# llama3-8b-8192 was deprecated by Groq in 2025 in favor of llama-3.1-8b-instant,
# so that's the default; override via .env if you want to pin a different model.
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")


def _extract_relevant_lines(query: str, text: str) -> str:
    query_lower = query.lower()
    fare_keywords = {"value": "Value", "flexi": "Flexi", "xtra": "Xtra"}
    detected_fare = None
    for keyword, label in fare_keywords.items():
        if keyword in query_lower:
            detected_fare = label
            break
    if not detected_fare:
        return text
    lines = text.split("\n")
    capture = False
    result = []
    for line in lines:
        if line.strip().startswith(detected_fare):
            capture = True
        elif any(line.strip().startswith(label) for label in fare_keywords.values()) and capture:
            break
        if capture:
            result.append(line)
    return "\n".join(result) if result else text


def _build_prompt(query: str, contexts: list[tuple[str, str]]) -> str:
    """Builds the CONTEXT + QUESTION portion only. SYSTEM_PROMPT is sent
    separately as a system message (Groq) or prepended (Ollama) so that the
    constraints aren't just one more sentence buried in a wall of text."""
    context_blocks: list[str] = []
    for source, text in contexts:
        filtered = _extract_relevant_lines(query, text)
        context_blocks.append(f"[Source: {source}]\n{filtered}")

    context_text = "\n\n".join(context_blocks)
    return (
        f"CONTEXT:\n{context_text}\n\n"
        f"CUSTOMER QUESTION: {query}\n"
        "ANSWER:"
    )


def _generate_with_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to a .env file in the project root, "
            "e.g. GROQ_API_KEY=gsk_..."
        )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 150,
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()

    data = response.json()
    try:
        return str(data["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected Groq response shape: {data}") from exc


def _generate_with_ollama(prompt: str) -> str:
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.1,
            "num_predict": 150,
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()

    data = response.json()
    if "response" not in data:
        raise ValueError("Ollama response JSON missing 'response' field.")
    return str(data["response"]).strip()


def _generate(prompt: str) -> str:
    if LLM_BACKEND == "ollama":
        return _generate_with_ollama(prompt)
    return _generate_with_groq(prompt)


def ask(query: str) -> dict:
    """
    Run retrieval + guarded generation for a user question.

    Args:
        query: User's question text.

    Returns:
        {
            "answer": str,
            "sources": [list of source filenames used],
            "is_relevant": bool
        }
    """
    retrieval = retrieve(query, k=4)

    # Collect unique sources while preserving first-seen order.
    sources = list(dict.fromkeys(chunk.source for chunk in retrieval.chunks if chunk.source))

    # Hard guardrail: if query is not relevant, do NOT call the LLM.
    if not retrieval.is_relevant:
        return {
            "answer": FALLBACK_MESSAGE,
            "sources": sources,
            "is_relevant": False,
        }

    contexts = [(chunk.source, chunk.text) for chunk in retrieval.chunks]
    prompt = _build_prompt(query=query, contexts=contexts)
    answer = _generate(prompt)

    return {
        "answer": answer,
        "sources": sources,
        "is_relevant": True,
    }

def _print_result(result: dict, elapsed_seconds: float) -> None:
    print("\nAnswer:")
    print(result["answer"])
    print("\nSources used:")
    if result["sources"]:
        for source in result["sources"]:
            print(f"- {source}")
    else:
        print("- (none)")
    print(f"\nIs relevant: {result['is_relevant']}")
    print(f"Response time: {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        question = " ".join(sys.argv[1:]).strip()
        started = time.perf_counter()
        result = ask(question)
        _print_result(result, time.perf_counter() - started)
    else:
        print("BlueBot ready. Type a question, or 'exit' to quit.")
        while True:
            question = input("\n> ").strip()
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue
            started = time.perf_counter()
            result = ask(question)
            _print_result(result, time.perf_counter() - started)
   