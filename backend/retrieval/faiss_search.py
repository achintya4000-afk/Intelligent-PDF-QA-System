
import json
from pathlib import Path

import faiss
import numpy as np

from backend.retrieval.embeddings import (
    generate_embeddings,
    generate_single_embedding
)


BASE_DIR = Path(__file__).resolve().parents[2]
INDEX_DIR = BASE_DIR / "data" / "indexes"


def create_faiss_index(document_id: str, chunks: list) -> int:
    """
    Generate embeddings and build a FAISS index for a document.

    Returns the number of indexed chunks.
    """

    if not document_id:
        raise ValueError("Document ID is required.")

    if not chunks:
        raise ValueError("At least one text chunk is required.")

    texts = [chunk["text"] for chunk in chunks]

    # Convert PDF chunks into embeddings
    embeddings = generate_embeddings(texts)

    dimension = embeddings.shape[1]

    # Exact cosine similarity search for normalized vectors
    index = faiss.IndexFlatIP(dimension)

    # Add vectors to the index
    index.add(np.ascontiguousarray(embeddings))

    # Save index under this document's directory
    document_dir = INDEX_DIR / document_id
    document_dir.mkdir(parents=True, exist_ok=True)

    index_path = document_dir / "faiss.index"
    faiss.write_index(index, str(index_path))

    return index.ntotal


def search_faiss(
    document_id: str,
    query: str,
    top_k: int = 5
) -> list:
    """
    Find PDF chunks that are semantically similar to a query.

    Returns matching chunks with page numbers and scores.
    """

    if not document_id:
        raise ValueError("Document ID is required.")

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    document_dir = INDEX_DIR / document_id
    index_path = document_dir / "faiss.index"
    chunks_path = document_dir / "chunks.json"

    if not index_path.exists():
        raise FileNotFoundError(
            "FAISS index not found for this document."
        )

    if not chunks_path.exists():
        raise FileNotFoundError(
            "Stored PDF chunks not found for this document."
        )

    # Load stored chunks
    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    # Load FAISS index
    index = faiss.read_index(str(index_path))

    if index.ntotal != len(chunks):
        raise ValueError(
            "FAISS index and stored chunks have different counts."
        )

    # Convert question into an embedding
    query_embedding = generate_single_embedding(query)

    query_vector = np.ascontiguousarray(
        query_embedding.reshape(1, -1),
        dtype=np.float32
    )

    # Search for the closest vectors
    k = min(top_k, index.ntotal)

    scores, indices = index.search(query_vector, k)

    results = []

    for score, chunk_index in zip(scores[0], indices[0]):

        if chunk_index < 0:
            continue

        chunk = chunks[int(chunk_index)]

        results.append({
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"],
            "text": chunk["text"],
            "score": float(score)
        })

    return results
