"""
Stage 4: Text Cleaning & Chunking
- Removes filler words, noise patterns
- Splits transcript into LangChain-compatible chunks
- Prepares structured input for the LLM pipeline
"""

import re
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Common filler words/phrases to remove
FILLER_PATTERNS = [
    r"\b(um|uh|er|ah|like|you know|i mean|basically|literally|actually|obviously|right)\b",
    r"\b(so so|kind of|sort of|you see|well well|hmm+|mm+|hm+)\b",
    r"\[.*?\]",          # Remove bracketed annotations like [laughter], [music]
    r"\(.*?\)",          # Remove parenthetical notes
    r"\.{3,}",           # Replace ellipsis chains
    r"\s{2,}",           # Collapse multiple spaces
]

SENTENCE_ENDINGS = re.compile(r"(?<=[.!?])\s+")


def clean_and_chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    remove_fillers: bool = True,
) -> List[str]:
    """
    Clean transcript text and split into overlapping chunks for LLM processing.

    Args:
        text: Raw transcript text
        chunk_size: Target chunk size in characters (approximates tokens)
        chunk_overlap: Number of characters to overlap between chunks
        remove_fillers: Whether to strip filler words

    Returns:
        List of cleaned text chunks
    """
    if not text or not text.strip():
        return []

    cleaned = _clean_text(text, remove_fillers)
    chunks = _split_into_chunks(cleaned, chunk_size, chunk_overlap)

    print(f"[TextProcessor] Cleaned text: {len(cleaned)} chars → {len(chunks)} chunks")
    return chunks


def _clean_text(text: str, remove_fillers: bool) -> str:
    """Remove noise and normalize the transcript text."""
    # Normalize unicode and encoding artifacts
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", " — ").replace("\u2013", "-")

    if remove_fillers:
        for pattern in FILLER_PATTERNS:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Fix sentence spacing
    text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)

    return text


def _split_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter.
    Falls back to manual splitting if LangChain is unavailable.
    """
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
            length_function=len,
        )
        docs = splitter.create_documents([text])
        return [doc.page_content for doc in docs]

    except ImportError:
        print("[TextProcessor] LangChain not found, using manual splitting.")
        return _manual_chunk(text, chunk_size, overlap)


def _manual_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Simple manual chunker as fallback."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Try to break at sentence boundary
        if end < len(text):
            for sep in [". ", "! ", "? ", "\n", " "]:
                idx = text.rfind(sep, start, end)
                if idx != -1:
                    end = idx + len(sep)
                    break

        chunks.append(text[start:end].strip())
        start = end - overlap if end - overlap > start else end

    return [c for c in chunks if c]
