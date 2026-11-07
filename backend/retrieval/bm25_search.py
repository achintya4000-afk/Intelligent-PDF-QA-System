import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


BASE_DIR = Path(__file__).resolve().parents[2]
INDEX_DIR = BASE_DIR / "data" / "indexes"


# Very common words carry no topic information. They are removed from the
# QUESTION only (never from the stored chunks) so that a query such as
# "What is the purpose of DNS?" is matched on "purpose" and "dns".
QUERY_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on",
    "at", "by", "for", "with", "about", "as", "from", "into", "between",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "can", "could", "should", "would", "will",
    "shall", "may", "might", "must",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "this", "that", "these", "those", "there", "here",
    "it", "its", "they", "them", "their", "you", "your", "i", "me", "my",
    "we", "our",
    "please", "explain", "describe", "tell", "give", "define", "list"
}


def tokenize_text(text: str) -> list:
    """
    Convert text into lowercase tokens.
    """

    if not isinstance(text, str):
        raise ValueError("Input must be a string.")

    return re.findall(r"\b\w+\b", text.lower())


def tokenize_query(text: str) -> list:
    """
    Convert a question into lowercase tokens without stopwords.
    """

    return [
        token
        for token in tokenize_text(text)
        if token not in QUERY_STOPWORDS
    ]


def search_bm25(
    document_id: str,
    query: str,
    top_k: int = 5
) -> list:
    """
    Search stored PDF chunks using BM25 keyword ranking.

    Only chunks that share at least one meaningful word with the
    question are returned. Chunks with no keyword overlap are NOT
    padded into the result list, because they would receive a
    reciprocal-rank contribution in hybrid search even though they
    are unrelated to the question.
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

    # Tokenize the user's question first (cheap) and stop early
    # when nothing meaningful is left.
    tokenized_query = tokenize_query(query)

    if not tokenized_query:
        return []

    # Tokenize all document chunks
    tokenized_chunks = [
        tokenize_text(chunk["text"])
        for chunk in chunks
    ]

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_chunks)

    # Calculate BM25 scores
    scores = bm25.get_scores(tokenized_query)

    # Keep only chunks that really contain a query word.
    # BM25Okapi can give zero or even slightly negative scores on very
    # small documents, so keyword overlap is checked directly instead of
    # relying on the score being positive.
    query_terms = set(tokenized_query)
    chunk_terms = [set(tokens) for tokens in tokenized_chunks]

    matching_indices = [
        index
        for index in range(len(chunks))
        if query_terms & chunk_terms[index]
    ]

    # Rank by BM25 score, using the number of matched words as a tie-breaker
    ranked_indices = sorted(
        matching_indices,
        key=lambda index: (
            scores[index],
            len(query_terms & chunk_terms[index])
        ),
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