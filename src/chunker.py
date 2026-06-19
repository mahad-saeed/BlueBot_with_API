"""
Text chunker for BlueBot RAG pipeline.

Reads raw policy documents from data/raw/, cleans scraped web artifacts,
and splits them into overlapping word-based chunks for embedding/retrieval.
"""

from __future__ import annotations

import re
from pathlib import Path

# Default location of raw text files (project_root/data/raw/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Chunking parameters (word-based)
CHUNK_SIZE_WORDS = 400
CHUNK_OVERLAP_WORDS = 50
MIN_WORDS_TO_CHUNK = 100  # Files shorter than this become a single chunk

# Lines copied from the Airblue site navigation bar (appear at top of scraped pages)
NAV_HEADER_LINES = frozenset(
    {
        "reservations",
        "travel deals",
        "destinations",
        "bluemiles",
        "login",
        "signup",
        "help",
        "contact centre",
        "welcome",
    }
)

# Footer / site chrome lines (appear at bottom of scraped pages)
FOOTER_LINES = frozenset(
    {
        "airblue",
        "our journey",
        "corporate information",
        "blue news",
        "careers",
        "services",
        "travel info",
        "flight status",
        "travel agents",
        "customer service",
        "contact us",
        "privacy policy",
        "legal terms & conditions",
        "health and travel guidelines",
        "passenger rights",
        "subscribe/unsubscribe to emails",
        "stay connected",
        "subscribe to our special offers",
        "subscribe",
        "download our app to manage flights",
        "google play",
        "app store",
        "airblue on facebook",
        "airblue on linkedin",
        "airblue on twitter",
        "airblue on instagram",
        "airblue on tiktok",
    }
)

# Login-form UI text that sometimes appears in scraped loyalty-program pages
LOGIN_FORM_LINES = frozenset(
    {
        "bluemiles login",
        "member id",
        "password",
        "remember me",
        "signin",
    }
)

# Breadcrumb navigation, e.g. "Home > Services > Baggage"
BREADCRUMB_PATTERN = re.compile(r"^[\w\s&/-]+(?:\s*>\s*[\w\s&/-]+)+$", re.IGNORECASE)

# Copyright / legal footer
COPYRIGHT_PATTERN = re.compile(r"©\s*airblue.*", re.IGNORECASE)

# Headquarters address line often repeated in footers
HQ_PATTERN = re.compile(r"airblue hdq:", re.IGNORECASE)


def _normalize_line(line: str) -> str:
    """Collapse internal whitespace and strip a single line."""
    return re.sub(r"\s+", " ", line).strip()


def _is_artifact_line(line: str) -> bool:
    """Return True if a line looks like site navigation or UI chrome."""
    normalized = _normalize_line(line).lower()

    if not normalized:
        return True

    # Standalone FAQ expand/collapse markers from scraped help pages
    if normalized == "+":
        return True

    if normalized in NAV_HEADER_LINES:
        return True

    if normalized in FOOTER_LINES:
        return True

    if normalized in LOGIN_FORM_LINES:
        return True

    if BREADCRUMB_PATTERN.match(normalized):
        return True

    if COPYRIGHT_PATTERN.search(normalized):
        return True

    if HQ_PATTERN.search(normalized):
        return True

    # Address fragment that only appears in the footer block
    if "ise towers" in normalized and "jinnah" in normalized:
        return True

    return False


def clean_text(raw_text: str) -> str:
    """
    Remove scraped navigation/footer artifacts and normalize whitespace.

    - Drops known nav, footer, breadcrumb, and form UI lines
    - Removes consecutive duplicate lines (repeated page titles/headers)
    - Collapses remaining text into a single normalized string
    """
    if not raw_text or not raw_text.strip():
        return ""

    cleaned_lines: list[str] = []
    previous_line: str | None = None

    for raw_line in raw_text.splitlines():
        line = _normalize_line(raw_line)
        if not line:
            continue

        if _is_artifact_line(line):
            continue

        # Skip immediately repeated headers/titles, e.g. "BlueMiles" twice in a row
        if previous_line is not None and line.lower() == previous_line.lower():
            continue

        cleaned_lines.append(line)
        previous_line = line

    # Join with spaces so chunk boundaries are not tied to original line breaks
    return re.sub(r"\s+", " ", " ".join(cleaned_lines)).strip()


def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """
    Split text into overlapping word-based chunks.

    Files with fewer than MIN_WORDS_TO_CHUNK words are returned as one chunk.
    Empty text returns an empty list.
    """
    words = text.split()
    if not words:
        return []

    # Very short documents are kept intact instead of being split further
    if len(words) < MIN_WORDS_TO_CHUNK:
        return [" ".join(words)]

    chunks: list[str] = []
    stride = chunk_size - overlap
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))

        if end >= len(words):
            break

        start += stride

    return chunks


def chunk_file(filepath: Path) -> list[dict[str, str]]:
    """
    Read, clean, and chunk a single .txt file.

    Returns a list of chunk dicts with keys: text, source, chunk_id.
    Empty files produce no chunks.
    """
    source_name = filepath.name
    stem = filepath.stem  # filename without extension, used in chunk_id

    try:
        raw_text = filepath.read_text(encoding="utf-8")
    except OSError:
        # Unreadable files are skipped rather than failing the whole batch
        return []

    cleaned = clean_text(raw_text)
    if not cleaned:
        return []

    text_chunks = split_into_chunks(cleaned)
    return [
        {
            "text": chunk_text,
            "source": source_name,
            "chunk_id": f"{stem}_{index}",
        }
        for index, chunk_text in enumerate(text_chunks)
    ]


def create_chunks(raw_dir: Path | str = RAW_DATA_DIR) -> list[dict[str, str]]:
    """
    Process every .txt file in raw_dir and return all chunks.

    Files are processed in sorted filename order for deterministic output.
    """
    raw_path = Path(raw_dir)
    if not raw_path.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {raw_path}")

    all_chunks: list[dict[str, str]] = []

    for txt_file in sorted(raw_path.glob("*.txt")):
        all_chunks.extend(chunk_file(txt_file))

    return all_chunks


if __name__ == "__main__":
    chunks = create_chunks()

    print(f"Total chunks created: {len(chunks)}")

    if chunks:
        print("\nSample (first chunk):")
        sample = chunks[0]
        print(f"  chunk_id: {sample['chunk_id']}")
        print(f"  source:   {sample['source']}")
        preview = sample["text"]
        if len(preview) > 500:
            preview = preview[:500] + "..."
        print(f"  text:     {preview}")
    else:
        print("\nNo chunks were created. Check that data/raw/ contains .txt files.")
