"""Highlight a glossary term on one PDF page, using real text positions.

The PDF viewer shows the browser's own PDF viewer, which cannot be searched
by a page script. Instead, this module finds the term's characters with
PyMuPDF and adds standard yellow "highlight" annotations to an in-memory copy
of the PDF. The uploaded file on disk is never modified.

Only characters that PyMuPDF reports for the page are highlighted, so the
rectangles always sit on the real text. A scanned page without a text layer
has no such characters, so nothing is highlighted there.
"""

import re
import unicodedata
from collections import OrderedDict
from pathlib import Path
from threading import Lock

import pymupdf


MAX_TERM_LENGTH = 200

# Longest all-capital term still treated as an abbreviation (matched exactly
# as written, so "IT" does not highlight the word "it").
MAX_ABBREVIATION_LENGTH = 6

HIGHLIGHT_COLOR = (1, 1, 0)  # yellow

# A few finished PDFs are kept so the "does it match?" request and the PDF
# request made right after it do not repeat the work.
_CACHE_SIZE = 3
_cache: "OrderedDict[tuple, tuple[bytes, int]]" = OrderedDict()
_cache_lock = Lock()

_DASHES = dict.fromkeys(
    map(ord, "‐‑‒–—―−"), "-"
)
_QUOTES = {
    ord("‘"): "'", ord("’"): "'", ord("‛"): "'",
    ord("“"): '"', ord("”"): '"', ord("‟"): '"',
}


def _normalize_char(char: str) -> str:
    """Normalises one PDF character (ligatures, dashes, quotes, spaces)."""
    if char == "­":  # soft hyphen: invisible in the PDF
        return ""

    char = unicodedata.normalize("NFKC", char)
    char = char.translate(_DASHES).translate(_QUOTES)

    return " " if char.isspace() else char


def _normalize_term(term: str) -> str:
    text = "".join(_normalize_char(char) for char in term)
    return re.sub(r"\s+", " ", text).strip()


def _page_characters(page: pymupdf.Page):
    """Returns the page text as parallel lists: characters and their boxes.

    Each box is (line_id, rect, rotated). Line and block breaks become a
    single space with no box, so a multi-word term can wrap onto a new line.
    """
    text: list[str] = []
    boxes: list[tuple | None] = []

    data = page.get_text("rawdict")
    line_id = 0

    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            line_id += 1
            direction = line.get("dir", (1, 0))
            rotated = abs(direction[1]) > 0.01

            for span in line.get("spans", []):
                for char in span.get("chars", []):
                    normalized = _normalize_char(char["c"])

                    for item in normalized:
                        text.append(item)
                        boxes.append(
                            (line_id, pymupdf.Rect(char["bbox"]), rotated)
                        )

            text.append(" ")
            boxes.append(None)

        text.append(" ")
        boxes.append(None)

    return text, boxes


def _term_candidates(term: str) -> list[str]:
    """The term itself, then (for "Name (ABBR)") the name and the abbreviation."""
    full = _normalize_term(term)

    if not full:
        return []

    candidates = [full]
    outside = _normalize_term(re.sub(r"\([^)]*\)", " ", full))
    inside = [
        _normalize_term(part)
        for part in re.findall(r"\(([^)]*)\)", full)
    ]

    for extra in [outside, *inside]:
        if extra and extra not in candidates:
            candidates.append(extra)

    return candidates


def _is_abbreviation(term: str) -> bool:
    letters = [char for char in term if char.isalpha()]
    compact = term.replace(" ", "")

    return (
        bool(letters)
        and not any(char.islower() for char in letters)
        and len(compact) <= MAX_ABBREVIATION_LENGTH
    )


def _build_pattern(term: str) -> re.Pattern:
    """Whole-term pattern: any whitespace between words, no partial words."""
    words = [re.escape(word) for word in term.split(" ")]
    body = r"\s+".join(words)

    abbreviation = _is_abbreviation(term)
    first, last = term[0], term[-1]

    # A trailing "s"/"es" is allowed so "network" also finds "networks".
    if last.isalpha():
        if abbreviation:
            body += "s?"
        elif len(term) >= 4:
            body += "(?:e?s)?"

    prefix = r"(?<!\w)" if (first.isalnum() or first == "_") else ""
    suffix = r"(?!\w)" if (last.isalnum() or last == "_") else ""

    flags = 0 if abbreviation else re.IGNORECASE

    return re.compile(f"{prefix}{body}{suffix}", flags)


def _rects_for_match(boxes, start: int, end: int) -> list[pymupdf.Rect]:
    """One rectangle per text line covered by the match (per character when
    the text is rotated, so the rectangle does not cover empty space)."""
    rects: list[pymupdf.Rect] = []
    current_line = None
    current_rect = None

    for box in boxes[start:end]:
        if box is None:
            continue

        line_id, rect, rotated = box

        if rotated:
            rects.append(pymupdf.Rect(rect))
            current_line, current_rect = None, None
            continue

        if line_id != current_line:
            if current_rect is not None:
                rects.append(current_rect)

            current_line, current_rect = line_id, pymupdf.Rect(rect)
        else:
            current_rect |= rect

    if current_rect is not None:
        rects.append(current_rect)

    return [rect for rect in rects if not rect.is_empty]


def find_term_rects(page: pymupdf.Page, term: str) -> list[pymupdf.Rect]:
    """All rectangles where the term appears on the page."""
    candidates = _term_candidates(term[:MAX_TERM_LENGTH])

    if not candidates:
        return []

    text, boxes = _page_characters(page)

    if not text:
        return []

    page_text = "".join(text)

    for candidate in candidates:
        rects: list[pymupdf.Rect] = []

        for match in _build_pattern(candidate).finditer(page_text):
            rects.extend(_rects_for_match(boxes, match.start(), match.end()))

        if rects:
            return rects

    return []


def highlight_term(
    pdf_path: Path, page_number: int, term: str
) -> tuple[bytes | None, int]:
    """Returns (PDF bytes with the term highlighted, number of matches).

    Returns (None, 0) when the term is not on the page or the page does not
    exist; the caller then serves the original, unchanged PDF.
    """
    stat = pdf_path.stat()
    key = (str(pdf_path), stat.st_mtime_ns, page_number, term)

    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]

    with pymupdf.open(pdf_path) as document:
        if page_number < 1 or page_number > document.page_count:
            return None, 0

        page = document[page_number - 1]
        rects = find_term_rects(page, term)

        if not rects:
            result: tuple[bytes | None, int] = (None, 0)
        else:
            for rect in rects:
                annotation = page.add_highlight_annot(rect)
                annotation.set_colors(stroke=HIGHLIGHT_COLOR)
                annotation.update()

            result = (document.tobytes(), len(rects))

    with _cache_lock:
        _cache[key] = result
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)

    return result
