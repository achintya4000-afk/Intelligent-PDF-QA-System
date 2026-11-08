"""
Optional Gemini Vision handwriting recogniser for DocuMind.

PaddleOCR remains responsible for printed OCR. This module asks Gemini to
identify and transcribe handwritten regions in the same rendered image.
Returned boxes are normalized to the image dimensions (0..1000).

If Gemini is not configured, the request fails, or usable boxes are not
returned, the caller can safely retain PaddleOCR output.
"""

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_client = None
_client_lock = threading.Lock()
_request_lock = threading.Lock()

# Temporary Gemini failures (rate limit, server error, timeout) are retried at
# most this many times after the first attempt, with exponential backoff:
# 2 s before the first retry and 4 s before the second.
_MAX_RETRIES = 2
_RETRY_BASE_DELAY_SECONDS = 2.0


def _is_retryable_error(error: Exception) -> bool:
    """
    True only for temporary failures: HTTP 429, 408 and 5xx responses, and
    timeouts. Permanent errors (invalid API key, invalid request, permission
    or model errors) are never retried.
    """
    code = getattr(error, "code", None)

    if isinstance(code, int) and not isinstance(code, bool):
        return code in (408, 429) or 500 <= code <= 599

    if isinstance(error, TimeoutError):
        return True

    # Timeout exceptions from the HTTP layer (for example a read timeout) do
    # not carry an HTTP status; recognise them by class name.
    return any(
        "timeout" in cls.__name__.lower()
        for cls in type(error).__mro__
    )


@dataclass(frozen=True)
class HandwritingLine:
    text: str
    score: float
    box: Optional[tuple]


_PROMPT = """
Inspect this document image and identify ONLY handwritten text regions.
Do not transcribe printed or typeset text.

Return valid JSON only, with this exact structure:
{
  "regions": [
    {
      "bbox": [x0, y0, x1, y1],
      "text": "faithful transcription of the handwritten region"
    }
  ]
}

Bounding box rules:
- Coordinates are integers from 0 to 1000, relative to the full image.
- bbox order is [left, top, right, bottom].
- The box must tightly surround the handwritten text region, not the whole page.
- Use one box per coherent handwritten paragraph or group of adjacent handwritten lines.
- Do not include printed text in a handwritten region.
- Preserve wording, numbers, punctuation and line order.
- Do not correct grammar or complete missing words.
- Use [unclear] for genuinely unreadable words.
- If there is no handwriting, return {"regions": []}.
"""


def _get_client_and_model():
    """Load Gemini configuration lazily so PaddleOCR still works on failure."""
    global _client

    try:
        from backend.config import GEMINI_API_KEY, GEMINI_MODEL
    except Exception as exc:
        raise RuntimeError("Gemini configuration is unavailable.") from exc

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    if _client is None:
        with _client_lock:
            if _client is None:
                from google import genai
                from google.genai import types

                # Google GenAI HttpOptions.timeout is expressed in milliseconds.
                # Keep the wait bounded so the extractor can fall back to PaddleOCR.
                try:
                    timeout_ms = int(os.getenv("GEMINI_TIMEOUT_MS", "60000"))
                except (TypeError, ValueError):
                    timeout_ms = 60000

                # Avoid an accidental zero/negative timeout from environment settings.
                timeout_ms = max(1000, timeout_ms)

                _client = genai.Client(
                    api_key=GEMINI_API_KEY,
                    http_options=types.HttpOptions(timeout=timeout_ms)
                )

    return _client, GEMINI_MODEL


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Gemini did not return valid JSON.")
        value = json.loads(text[start:end + 1])

    if not isinstance(value, dict):
        raise ValueError("Gemini JSON response must be an object.")
    return value


def recognize_handwriting_regions(image_bgr: np.ndarray) -> list:
    """
    Return handwritten regions as HandwritingLine objects.

    The image must be a uint8 BGR array, matching the image sent to PaddleOCR.
    Each returned box is in pixels of that same image.
    """
    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3:
        raise ValueError("Expected a three-channel image array.")
    if image_bgr.shape[2] != 3 or image_bgr.size == 0:
        raise ValueError("Expected a non-empty BGR image.")

    enabled = os.getenv("GEMINI_HANDWRITING_OCR", "true").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return []

    client, model = _get_client_and_model()

    # Pillow is already a standard dependency of the Google/Paddle stack.
    rgb = image_bgr[:, :, ::-1]
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    from google.genai import types

    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
    config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json"
    )

    # Serialise calls: the same FastAPI process may receive concurrent uploads.
    # The lock is held only while a request is in flight, not while waiting
    # between retries.
    response = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            with _request_lock:
                logger.info(
                    "Sending handwriting region to Gemini model %s "
                    "(timeout: %s ms).",
                    model,
                    os.getenv("GEMINI_TIMEOUT_MS", "60000")
                )
                response = client.models.generate_content(
                    model=model,
                    contents=[_PROMPT, image_part],
                    config=config
                )
            break
        except Exception as error:
            if attempt >= _MAX_RETRIES or not _is_retryable_error(error):
                raise

            delay = _RETRY_BASE_DELAY_SECONDS * (2 ** attempt)
            logger.warning(
                "Gemini handwriting request failed with a temporary error "
                "(%r); retry %s of %s in %.0f s.",
                error,
                attempt + 1,
                _MAX_RETRIES,
                delay
            )
            time.sleep(delay)

    payload = _extract_json(response.text)
    regions = payload.get("regions", [])
    if not isinstance(regions, list):
        raise ValueError("Gemini 'regions' field must be a list.")

    height, width = image_bgr.shape[:2]
    recognised = []

    for item in regions:
        if not isinstance(item, dict):
            continue

        text = str(item.get("text", "")).strip()
        bbox = item.get("bbox")

        if not text or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        try:
            x0n, y0n, x1n, y1n = [float(value) for value in bbox]
        except (TypeError, ValueError):
            continue

        if not np.isfinite([x0n, y0n, x1n, y1n]).all():
            continue
        if not (0 <= x0n < x1n <= 1000 and 0 <= y0n < y1n <= 1000):
            continue

        x0 = max(0, min(width, round(x0n * width / 1000)))
        y0 = max(0, min(height, round(y0n * height / 1000)))
        x1 = max(0, min(width, round(x1n * width / 1000)))
        y1 = max(0, min(height, round(y1n * height / 1000)))

        if x1 <= x0 or y1 <= y0:
            continue

        recognised.append(
            HandwritingLine(
                text=text,
                score=1.0,  # Gemini does not provide a comparable OCR confidence.
                box=(float(x0), float(y0), float(x1), float(y1))
            )
        )

    logger.info("Gemini identified %s handwritten region(s).", len(recognised))
    return recognised


def _overlap_fraction(line_box: tuple, region_box: tuple) -> float:
    """Intersection area divided by the PaddleOCR line-box area."""
    lx0, ly0, lx1, ly1 = line_box
    rx0, ry0, rx1, ry1 = region_box

    line_area = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
    if line_area <= 0:
        return 0.0

    intersection_width = max(0.0, min(lx1, rx1) - max(lx0, rx0))
    intersection_height = max(0.0, min(ly1, ry1) - max(ly0, ry0))
    return (intersection_width * intersection_height) / line_area


def remove_printed_lines_overlapping_handwriting(
    paddle_lines: list,
    handwriting_lines: list,
    overlap_threshold: float = 0.25
) -> list:
    """
    Remove PaddleOCR lines substantially covered by Gemini handwriting boxes.
    Lines without geometry are retained to avoid discarding uncertain content.
    """
    if not handwriting_lines:
        return list(paddle_lines)

    handwriting_boxes = [
        item.box for item in handwriting_lines if item.box is not None
    ]
    kept = []

    for line in paddle_lines:
        if line.box is None:
            kept.append(line)
            continue

        overlaps_handwriting = any(
            _overlap_fraction(line.box, box) >= overlap_threshold
            for box in handwriting_boxes
        )
        if not overlaps_handwriting:
            kept.append(line)

    return kept
