"""
Evidence check used by the Auto response mode.

Hybrid retrieval always returns the "best" chunks, even when none of them
is related to the question. This module decides whether the retrieved
chunks are worth sending to Gemini as PDF evidence.

Why the decision does not use RRF or BM25 scores
------------------------------------------------
* RRF score = sum(1 / (k + rank)). It depends only on ranks, so the top
  chunk of an unrelated question scores exactly like the top chunk of a
  perfect match.
* BM25 scores depend on the size and vocabulary of the document and can be
  zero or negative in small PDFs, so they have no universal scale.
* FAISS scores are cosine similarities of normalised vectors. They are
  bounded (-1..1) and comparable between questions, so they are the main
  semantic signal.

Signals
-------
1. Best FAISS cosine score among the retrieved chunks.
2. Lexical coverage: the largest share of the question's content words
   (stop words removed, like BM25 does) found inside one retrieved chunk.
   It rescues exact-term questions such as "What is OSPF?", where a short
   acronym gives a modest cosine score although the PDF explains it.

This is only a first filter. The final decision still belongs to the
grounded answer step: if Gemini abstains or returns no valid citation,
the chat service falls back to a General AI answer.
"""

import logging

from backend.config import (
    AUTO_MIN_SEMANTIC_SCORE,
    AUTO_CONFIDENT_SEMANTIC_SCORE,
    AUTO_MIN_LEXICAL_COVERAGE
)
from backend.retrieval.bm25_search import tokenize_query, tokenize_text


logger = logging.getLogger("uvicorn.error")


def _normalise_term(term: str) -> str:
    """
    Crude plural / verb-ending normalisation ("protocols" -> "protocol").

    Only used to compare question words with chunk words.
    """

    if len(term) > 5 and term.endswith("ing"):
        return term[:-3]

    if len(term) > 4 and term.endswith("es"):
        return term[:-2]

    if len(term) > 3 and term.endswith("ed"):
        return term[:-2]

    if len(term) > 3 and term.endswith("s"):
        return term[:-1]

    return term


def _best_semantic_score(chunks: list):
    """
    Highest FAISS cosine score among the chunks, or None when no chunk
    came from the semantic search.
    """

    scores = [
        chunk["faiss_score"]
        for chunk in chunks
        if isinstance(chunk.get("faiss_score"), (int, float))
    ]

    return max(scores) if scores else None


def _best_lexical_coverage(query: str, chunks: list) -> float:
    """
    Largest share of the question's content words found in one chunk.
    """

    query_terms = {
        _normalise_term(term)
        for term in tokenize_query(query)
    }

    if not query_terms:
        return 0.0

    best = 0.0

    for chunk in chunks:
        text = chunk.get("text")

        if not isinstance(text, str):
            continue

        chunk_terms = {
            _normalise_term(term)
            for term in tokenize_text(text)
        }

        best = max(
            best,
            len(query_terms & chunk_terms) / len(query_terms)
        )

    return best


def assess_retrieval_evidence(query: str, chunks: list) -> dict:
    """
    Decide whether retrieved chunks are relevant enough to try a
    PDF-grounded answer.

    Returns:
        {
            "sufficient": bool,
            "reason": str,
            "semantic_score": float | None,
            "lexical_coverage": float
        }
    """

    if not isinstance(chunks, list) or not chunks:
        return {
            "sufficient": False,
            "reason": "no_chunks",
            "semantic_score": None,
            "lexical_coverage": 0.0
        }

    semantic_score = _best_semantic_score(chunks)
    lexical_coverage = _best_lexical_coverage(query, chunks)

    keywords_match = lexical_coverage >= AUTO_MIN_LEXICAL_COVERAGE

    if semantic_score is None:
        # Only keyword search found something.
        sufficient = keywords_match
        reason = "keyword_match" if sufficient else "keyword_match_too_weak"

    elif semantic_score < AUTO_MIN_SEMANTIC_SCORE:
        sufficient = False
        reason = "semantic_score_too_low"

    elif semantic_score >= AUTO_CONFIDENT_SEMANTIC_SCORE:
        sufficient = True
        reason = "strong_semantic_match"

    else:
        # In-between range: require an exact keyword match as well.
        sufficient = keywords_match
        reason = (
            "semantic_and_keyword_match"
            if sufficient
            else "semantic_match_without_keywords"
        )

    # Written to the backend terminal to help tune the thresholds.
    logger.info(
        "Auto evidence check: sufficient=%s reason=%s "
        "semantic_score=%s lexical_coverage=%.2f",
        sufficient,
        reason,
        "n/a" if semantic_score is None else f"{semantic_score:.3f}",
        lexical_coverage
    )

    return {
        "sufficient": sufficient,
        "reason": reason,
        "semantic_score": semantic_score,
        "lexical_coverage": lexical_coverage
    }
