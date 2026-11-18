"""
Google Search grounding helpers for general AI answers.

This module only extracts metadata returned by Gemini. It never scrapes
websites or fabricates source URLs.
"""

import re
from urllib.parse import urlparse


# Search is triggered by an explicit request for fresh external information,
# not by isolated words or a year that may simply appear in a PDF question.
CURRENT_INFO_PATTERNS = (
    r"\b(latest|newest|breaking|upcoming|recent developments|recent news|"
    r"latest developments|latest news|latest update|current events|"
    r"current status|as of now|right now|today's news|today news|"
    r"this week'?s news|this month's news|this year'?s updates|"
    r"who is the current|who's the current|what is the current (?:president|prime minister|ceo|leader|price|rate|version|weather|exchange rate)|"
    r"current (?:price|exchange rate|weather|version|ceo|prime minister|president))\b",
    r"\b(what happened|what's happening|whats happening|"
    r"what is happening|who won|who is winning|who currently leads)\b",
    r"\b(what time is it|what is the current time|what is the time in|current time in|"
    r"time in [a-z][a-z .'-]*|today's date|what date is it today)\b",
    r"\b(202[5-9]|203[0-9])\b.*\b(latest|recent|developments|news|updates)\b",
)
_DOCUMENT_CONTEXT_PATTERN = re.compile(
    r"\b(according to (?:the |my |this |your )?(?:pdf|document|report|file|paper)|"
    r"in (?:the |this |my |your |uploaded )?(?:pdf|document|report|file|paper)|"
    r"mentioned (?:in|by) (?:the |this |my |your )?(?:pdf|document|report|file|paper)|"
    r"discussed (?:in|by) (?:the |this |my |your )?(?:pdf|document|report|file|paper)|"
    r"from (?:the |this |my |your |uploaded )?(?:pdf|document|report|file|paper))\b",
    re.IGNORECASE,
)
_NON_CURRENT_CONTEXT_PATTERN = re.compile(
    r"\b(time complexity|time limit|time period|time complexity of|"
    r"date mentioned|date stated|current limitations?)\b",
    re.IGNORECASE,
)


def question_needs_current_information(question: str) -> bool:
    """Detect explicit freshness/location intent without matching PDF context."""
    text = re.sub(r"\s+", " ", (question or "").strip().lower())
    if not text:
        return False
    if _DOCUMENT_CONTEXT_PATTERN.search(text) or _NON_CURRENT_CONTEXT_PATTERN.search(text):
        return False
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in CURRENT_INFO_PATTERNS)


def _field(value, name, default=None):
    """Read a field from either an SDK object or a dictionary."""

    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _is_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except (TypeError, ValueError):
        return False


def extract_grounding_metadata(response) -> dict:
    """
    Extract Google Search sources, inline citation support and the official
    Search Suggestion HTML from a Gemini response.

    Source URLs and suggestion HTML are passed through only from Gemini's
    grounding metadata; this module never invents or scrapes sources.
    """

    candidates = _field(response, "candidates", []) or []
    candidate = candidates[0] if candidates else None
    metadata = _field(candidate, "grounding_metadata") if candidate else None

    if metadata is None:
        metadata = _field(candidate, "groundingMetadata") if candidate else None

    chunks = _field(metadata, "grounding_chunks", None)
    if chunks is None:
        chunks = _field(metadata, "groundingChunks", []) or []

    sources = []
    seen_urls = {}
    raw_chunk_source_ids = []

    for chunk in chunks:
        web = _field(chunk, "web")
        if web is None:
            raw_chunk_source_ids.append(None)
            continue

        uri = _field(web, "uri") or _field(web, "url")
        title = _field(web, "title") or "Web source"

        if not isinstance(uri, str) or not _is_http_url(uri):
            raw_chunk_source_ids.append(None)
            continue

        normalized = uri.strip()
        if normalized in seen_urls:
            raw_chunk_source_ids.append(seen_urls[normalized])
            continue

        source_id = f"W{len(sources) + 1}"
        seen_urls[normalized] = source_id
        raw_chunk_source_ids.append(source_id)
        sources.append({
            "source_id": source_id,
            "title": str(title).strip()[:300] or "Web source",
            "url": normalized
        })

    queries = _field(metadata, "web_search_queries", None)
    if queries is None:
        queries = _field(metadata, "webSearchQueries", []) or []

    search_queries = [
        str(query).strip()[:500]
        for query in queries
        if isinstance(query, str) and query.strip()
    ]

    supports = _field(metadata, "grounding_supports", None)
    if supports is None:
        supports = _field(metadata, "groundingSupports", []) or []

    inline_citations = []
    for support in supports:
        segment = _field(support, "segment")
        segment_text = _field(segment, "text") if segment else None
        indices = _field(support, "grounding_chunk_indices", None)
        if indices is None:
            indices = _field(support, "groundingChunkIndices", []) or []

        source_ids = []
        for index in indices:
            if isinstance(index, int) and 0 <= index < len(raw_chunk_source_ids):
                source_id = raw_chunk_source_ids[index]
                if source_id and source_id not in source_ids:
                    source_ids.append(source_id)

        if isinstance(segment_text, str) and segment_text.strip() and source_ids:
            inline_citations.append({
                "text": segment_text,
                "source_ids": source_ids
            })

    search_entry_point = _field(metadata, "search_entry_point", None)
    if search_entry_point is None:
        search_entry_point = _field(metadata, "searchEntryPoint", None)

    rendered_content = _field(search_entry_point, "rendered_content")
    if rendered_content is None:
        rendered_content = _field(search_entry_point, "renderedContent")

    return {
        "web_sources": sources,
        "search_queries": search_queries,
        "web_search_used": bool(search_queries or sources),
        "inline_web_citations": inline_citations,
        "search_suggestion_html": (
            rendered_content if isinstance(rendered_content, str) else ""
        )
    }


def add_inline_web_citations(answer: str, citations: list) -> str:
    """Append grounded [W#] labels only after exact, ordered segment matches.

    We intentionally do not retry from the beginning when a segment is not
    found after the previous match; this avoids attaching duplicate phrases
    to the wrong occurrence. Labels are placed after trailing Markdown
    emphasis delimiters and punctuation, not inside the matched text.
    """
    if not isinstance(answer, str) or not answer or not isinstance(citations, list):
        return answer

    insertions = []
    search_from = 0
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        segment = citation.get("text")
        source_ids = citation.get("source_ids", [])
        if not isinstance(segment, str) or not segment.strip():
            continue
        position = answer.find(segment, search_from)
        if position < 0:
            continue
        end = position + len(segment)
        # Keep closing Markdown emphasis/code delimiters and punctuation
        # attached to the text before adding a citation label.
        while end < len(answer) and answer[end] in "*_`":
            end += 1
        while end < len(answer) and answer[end] in ".,;:!?":
            end += 1
        labels = " ".join(
            f"[{source_id}]" for source_id in source_ids
            if isinstance(source_id, str) and re.fullmatch(r"W\d+", source_id)
        )
        if labels:
            insertions.append((end, f" {labels}"))
            search_from = end

    for position, label_text in sorted(insertions, reverse=True):
        answer = answer[:position] + label_text + answer[position:]
    return answer
