
from backend.retrieval.faiss_search import search_faiss
from backend.retrieval.bm25_search import search_bm25


def reciprocal_rank_fusion(
    result_lists: list,
    k: int = 60
) -> list:
    """
    Combine multiple ranked retrieval result lists using RRF.

    RRF score = sum(1 / (k + rank))

    Ranks start from 1.
    """

    if not isinstance(result_lists, list):
        raise ValueError("Result lists must be a list.")

    if k <= 0:
        raise ValueError("RRF constant must be greater than zero.")

    fused_results = {}

    for results in result_lists:

        seen_chunk_ids = set()

        for rank, result in enumerate(results, start=1):

            chunk_id = result.get("chunk_id")

            if chunk_id is None:
                raise ValueError("Every result must have a chunk_id.")

            if chunk_id in seen_chunk_ids:
                raise ValueError(
                    f"Duplicate chunk ID found: {chunk_id}"
                )

            seen_chunk_ids.add(chunk_id)

            if chunk_id not in fused_results:

                fused_results[chunk_id] = {
                    "chunk_id": chunk_id,
                    "page": result["page"],
                    "text": result["text"],
                    "faiss_score": None,
                    "bm25_score": None,
                    "rrf_score": 0.0
                }

            # Identify the retrieval source
            source = result.get("source")

            if source == "faiss":
                fused_results[chunk_id]["faiss_score"] = result["score"]

            elif source == "bm25":
                fused_results[chunk_id]["bm25_score"] = result["score"]

            # Add reciprocal rank contribution
            fused_results[chunk_id]["rrf_score"] += (
                1.0 / (k + rank)
            )

    # Sort by final RRF score
    ranked_results = sorted(
        fused_results.values(),
        key=lambda result: result["rrf_score"],
        reverse=True
    )

    return ranked_results


def hybrid_search(
    document_id: str,
    query: str,
    top_k: int = 5,
    candidate_k: int = 10,
    rrf_k: int = 60
) -> list:
    """
    Combine FAISS semantic search and BM25 keyword search.
    """

    if not document_id:
        raise ValueError("Document ID is required.")

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if candidate_k <= 0:
        raise ValueError("candidate_k must be greater than zero.")

    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than zero.")

    # Retrieve candidates using FAISS
    faiss_results = search_faiss(
        document_id,
        query,
        top_k=candidate_k
    )

    # Retrieve candidates using BM25
    bm25_results = search_bm25(
        document_id,
        query,
        top_k=candidate_k
    )

    # Mark the source of each retrieval result
    for result in faiss_results:
        result["source"] = "faiss"

    for result in bm25_results:
        result["source"] = "bm25"

    # Combine both rankings using RRF
    fused_results = reciprocal_rank_fusion(
        [faiss_results, bm25_results],
        k=rrf_k
    )

    return fused_results[:top_k]
