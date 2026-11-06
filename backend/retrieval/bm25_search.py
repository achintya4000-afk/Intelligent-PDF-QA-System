
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


BASE_DIR = Path(__file__).resolve().parents[2]
INDEX_DIR = BASE_DIR / "data" / "indexes"


def tokenize_text(text: str) -> list:
    """
    Convert text into lowercase tokens.
    """

    if not isinstance(text, str):
        raise ValueError("Input must be a string.")

    return re.findall(r"\b\w+\b", text.lower())


def search_bm25(
    document_id: str,
    query: str,
    top_k: int = 5
) -> list:
    """
    Search stored PDF chunks using BM25 keyword ranking.

    Returns matching chunks with BM25 scores.
    """

    if not document_id:
        raise ValueError("Document ID is required.")

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    chunks_path = INDEX_DIR / document_id / "chunks.json"

    if not chunks_path.exists():
        raise FileNotFoundError(
            "Stored PDF chunks not found for this document."
        )

    # Load document chunks
    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        return []

    # Tokenize all document chunks
    tokenized_chunks = [
        tokenize_text(chunk["text"])
        for chunk in chunks
    ]

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_chunks)

    # Tokenize the user's question
    tokenized_query = tokenize_text(query)

    if not tokenized_query:
        return []

    # Calculate BM25 scores
    scores = bm25.get_scores(tokenized_query)

    # Rank chunks by descending score
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True
    )

    results = []

    for index in ranked_indices[:top_k]:

        results.append({
            "chunk_id": chunks[index]["chunk_id"],
            "page": chunks[index]["page"],
            "text": chunks[index]["text"],
            "score": float(scores[index])
        })

    return results
