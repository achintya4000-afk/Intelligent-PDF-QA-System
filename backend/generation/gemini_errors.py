"""Helpers for classifying transient Gemini API failures."""

import re


def gemini_error_status(error: BaseException) -> int | None:
    """Extract an HTTP status from Google GenAI exception text safely."""
    text = str(error)
    match = re.search(r"(?:\b|HTTP\s*)(429|503|504)\b", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    # Google GenAI exceptions often include RESOURCE_EXHAUSTED / UNAVAILABLE.
    upper = text.upper()
    if "RESOURCE_EXHAUSTED" in upper or "TOO MANY REQUESTS" in upper:
        return 429
    if "SERVICE_UNAVAILABLE" in upper or "UNAVAILABLE" in upper:
        return 503
    if "GATEWAY TIMEOUT" in upper:
        return 504
    return None


def is_quota_error(error: BaseException) -> bool:
    return gemini_error_status(error) == 429


def is_retryable_transient_error(error: BaseException) -> bool:
    return gemini_error_status(error) in {503, 504}
