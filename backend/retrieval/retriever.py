
from backend.retrieval.hybrid_search import hybrid_search


def retrieve_relevant_chunks(
    document_id: str,
    query: str,
    top_k: int = 5,
    candidate_k: int = 10,
    rrf_k: int = 60
) -> list:
    """
    Retrieve the most relevant PDF chunks using hybrid search.

    Combines:
    1. FAISS semantic search
    2. BM25 keyword search
    3. Reciprocal Rank Fusion

    Returns ranked chunks with their page numbers,
    individual retrieval scores, and RRF scores.
    """

    # Validate document ID
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("A valid document ID is required.")

    # Validate query
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    # Validate retrieval parameters
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    if (
        not isinstance(candidate_k, int)
        or isinstance(candidate_k, bool)
        or candidate_k <= 0
    ):
        raise ValueError("candidate_k must be a positive integer.")

    if not isinstance(rrf_k, int) or isinstance(rrf_k, bool) or rrf_k <= 0:
        raise ValueError("rrf_k must be a positive integer.")

    # Perform hybrid retrieval
    results = hybrid_search(
        document_id=document_id,
        query=query.strip(),
        top_k=top_k,
        candidate_k=candidate_k,
        rrf_k=rrf_k
    )

    return results
