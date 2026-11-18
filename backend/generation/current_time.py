"""
Timezone-aware current date/time answers for Auto and Only AI modes.
"""

import os
import re
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
except ImportError:  # Python versions before zoneinfo (not expected on supported Python)
    ZoneInfo = None


DEFAULT_TIMEZONE = "Asia/Kolkata"
_IST_FALLBACK = timezone(timedelta(hours=5, minutes=30), name="IST")


def get_timezone_name() -> str:
    """Return a valid configured timezone, defaulting to Asia/Kolkata."""
    requested = (os.getenv("DOCUMIND_TIMEZONE") or DEFAULT_TIMEZONE).strip() or DEFAULT_TIMEZONE
    if ZoneInfo is None:
        return DEFAULT_TIMEZONE
    try:
        ZoneInfo(requested)
        return requested
    except Exception:
        return DEFAULT_TIMEZONE


def get_ist_now() -> datetime:
    """Return the current time in the configured timezone.

    Asia/Kolkata remains the default. If zoneinfo data is unavailable,
    retain the historical fixed +05:30 fallback for the default timezone.
    """
    timezone_name = get_timezone_name()
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo(timezone_name))
        except Exception:
            pass
    if timezone_name == DEFAULT_TIMEZONE:
        return datetime.now(_IST_FALLBACK)
    # An invalid override should not crash chat; fall back to the documented default.
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo(DEFAULT_TIMEZONE))
        except Exception:
            pass
    return datetime.now(_IST_FALLBACK)


def is_current_time_question(question: str) -> bool:
    """Recognise direct requests for the server's current local date/time.

    Full-string patterns deliberately avoid intercepting phrases such as
    "time complexity", "time limit", or dates mentioned in a document.
    Location-specific questions are left to the AI/search path.
    """
    text = re.sub(r"\s+", " ", (question or "").strip().lower())
    text = re.sub(r"[?!.]+$", "", text).strip()
    if not text:
        return False

    # A named location requires that location's clock, not the server clock.
    if re.search(r"\b(?:in|at)\s+[a-z][a-z .'-]*$", text) and not re.search(
        r"\bat the (?:moment|present)$", text
    ):
        return False

    patterns = (
        r"(?:please\s+)?what time is it(?: now| right now| at the moment)?",
        r"(?:please\s+)?what's the time(?: now| right now)?",
        r"(?:please\s+)?whats the time(?: now| right now)?",
        r"(?:please\s+)?what time is it at the moment",
        r"(?:please\s+)?what is the current time",
        r"(?:please\s+)?what is the time in [a-z][a-z .'-]*",
        r"(?:please\s+)?what's the current time",
        r"(?:please\s+)?whats the current time",
        r"(?:please\s+)?tell me the current time",
        r"(?:please\s+)?tell me the time(?: now)?",
        r"(?:please\s+)?show me the current time",
        r"(?:please\s+)?what is the time now",
        r"(?:please\s+)?what is today's date",
        r"(?:please\s+)?what is todays date",
        r"(?:please\s+)?what is the current date",
        r"(?:please\s+)?what's today's date",
        r"(?:please\s+)?whats today's date",
        r"(?:please\s+)?what date is it(?: today)?",
        r"(?:please\s+)?what day is it(?: today)?",
        r"(?:please\s+)?tell me today's date",
        r"(?:please\s+)?tell me the date",
        r"(?:please\s+)?what is today's day",
        r"(?:please\s+)?what is the date today",
    )
    return any(re.fullmatch(pattern, text) for pattern in patterns)


def build_current_time_answer() -> dict:
    """Build a deterministic current date/time response (not model-generated)."""

    now = get_ist_now()
    date_text = now.strftime("%A, %d %B %Y").replace(" 0", " ")
    time_text = now.strftime("%I:%M:%S %p").lstrip("0")
    iso_text = now.isoformat(timespec="seconds")
    timezone_name = get_timezone_name()
    timezone_label = now.tzname() or timezone_name
    location_label = "in India" if timezone_name == DEFAULT_TIMEZONE else f"in {timezone_name}"

    return {
        "answer": (
            f"The current date {location_label} is **{date_text}**.\n\n"
            f"The current time is **{time_text} {timezone_label}** "
            f"({timezone_name})."
        ),
        "sources": [],
        "web_sources": [],
        "search_queries": [],
        "web_search_used": False,
        "supported": False,
        "current_time_used": True,
        "current_time_iso": iso_text
    }
