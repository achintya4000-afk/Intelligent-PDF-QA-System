"""
Local OCR engine for DocuMind.

Wraps PaddleOCR (PP-OCRv5, CPU only) so that the PDF extractor can recognise
text in scanned and handwritten pages. No cloud service, API key or billing
is involved. Models are downloaded once by PaddleOCR (PaddleX) into its local
cache (by default "~/.paddlex/official_models") and are reused afterwards.

The OCR pipeline is created lazily on first use and then reused for every
page and every document, so the models are loaded only once per process.

Optional environment variables (all have safe defaults):

    OCR_ENABLED             true/false  Turn OCR on or off (default: true)
    OCR_MODE                auto | always
                              auto   = OCR scanned pages, and image regions or
                                       annotations on pages that also have
                                       digital text (default)
                              always = OCR every page completely (slowest, most
                                       thorough; also catches vector "ink")
    OCR_DETECTION_MODEL     detection model   (default: PP-OCRv5_mobile_det)
    OCR_RECOGNITION_MODEL   recognition model (default: en_PP-OCRv5_mobile_rec)
    OCR_CPU_THREADS         CPU threads for inference (default: half the logical
                            processors, between 2 and 8)
    OCR_ENABLE_MKLDNN       true/false  oneDNN acceleration (default: false)
    OCR_RENDER_DPI          page render resolution (default: 200)
    OCR_MAX_IMAGE_SIDE      longest rendered side in pixels (default: 2600)

    OCR_MIN_CONFIDENCE      0..1  Drop recognised lines scoring below this
                            (default: 0.15). See the note below.
    OCR_DET_BOX_THRESH      0..1  Optional. PaddleOCR text_det_box_thresh
                            (library default 0.6). Lower values (0.4-0.5) find
                            fainter text such as light pencil handwriting, at
                            the cost of more false boxes.
    OCR_DET_UNCLIP_RATIO    0.5..4  Optional. PaddleOCR text_det_unclip_ratio.
                            Higher values widen the detected boxes, which can
                            help loose handwriting.

    OCR_MIN_IMAGE_AREA_RATIO  0..0.5 Smallest image (share of the page) that is
                              OCR'd on a page that already has digital text
                              (default: 0.01)

About OCR_MIN_CONFIDENCE: every recognised line has a confidence score. A
high threshold removes junk (watermarks, logos, text in scripts the English
model cannot read) but also removes real handwriting, because recognition
scores for handwriting are much lower than for print. A low threshold keeps
more handwriting but lets more junk into the index. 0.15 only removes lines
the model is nearly sure it cannot read. Use 0.0 to keep everything for
handwriting-heavy documents, or 0.5 for clean printed documents. Dropped lines
are always reported in the log, and a scanned page whose lines would ALL be
dropped is kept instead of becoming empty.

Run "python -m backend.pdf_processing.ocr_engine" once to download and
warm up the models before the first upload.
"""

import logging
import os
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

# PaddleX checks model-hoster connectivity on start-up, which only slows
# start-up down. PaddleX treats any non-empty value as "skip the check". The
# user can still override this in the environment.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

logger = logging.getLogger(__name__)


def _bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)

    if raw_value is None or not raw_value.strip():
        return default

    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw_value = os.getenv(name)

    if raw_value is None or not raw_value.strip():
        return default

    try:
        value = int(raw_value)
    except ValueError:
        return default

    return value if minimum <= value <= maximum else default


def _float_env(name: str, default: float, minimum: float, maximum: float) -> float:
    raw_value = os.getenv(name)

    if raw_value is None or not raw_value.strip():
        return default

    try:
        value = float(raw_value)
    except ValueError:
        return default

    return value if minimum <= value <= maximum else default


def _optional_float_env(name: str, minimum: float, maximum: float):
    """A float setting that is only passed to PaddleOCR when it is set."""

    raw_value = os.getenv(name)

    if raw_value is None or not raw_value.strip():
        return None

    try:
        value = float(raw_value)
    except ValueError:
        return None

    return value if minimum <= value <= maximum else None


OCR_ENABLED = _bool_env("OCR_ENABLED", True)

_mode = (os.getenv("OCR_MODE", "") or "auto").strip().lower()
OCR_MODE = _mode if _mode in {"auto", "always"} else "auto"

OCR_DETECTION_MODEL = (
    os.getenv("OCR_DETECTION_MODEL", "").strip() or "PP-OCRv5_mobile_det"
)
OCR_RECOGNITION_MODEL = (
    os.getenv("OCR_RECOGNITION_MODEL", "").strip() or "en_PP-OCRv5_mobile_rec"
)

OCR_CPU_THREADS = _int_env(
    "OCR_CPU_THREADS",
    default=max(2, min(8, (os.cpu_count() or 4) // 2)),
    minimum=1,
    maximum=64
)
OCR_ENABLE_MKLDNN = _bool_env("OCR_ENABLE_MKLDNN", False)

OCR_RENDER_DPI = _int_env("OCR_RENDER_DPI", 200, 72, 400)
OCR_MAX_IMAGE_SIDE = _int_env("OCR_MAX_IMAGE_SIDE", 2600, 800, 6000)

OCR_MIN_CONFIDENCE = _float_env("OCR_MIN_CONFIDENCE", 0.15, 0.0, 1.0)
OCR_DET_BOX_THRESH = _optional_float_env("OCR_DET_BOX_THRESH", 0.0, 1.0)
OCR_DET_UNCLIP_RATIO = _optional_float_env("OCR_DET_UNCLIP_RATIO", 0.5, 4.0)

OCR_MIN_IMAGE_AREA_RATIO = _float_env("OCR_MIN_IMAGE_AREA_RATIO", 0.01, 0.0, 0.5)


class OCRUnavailableError(RuntimeError):
    """The OCR engine is disabled, not installed or could not start."""


class OCRInferenceError(RuntimeError):
    """OCR failed while reading one page image."""


@dataclass(frozen=True)
class OCRLine:
    """One recognised text line with its box in image pixels."""

    text: str
    score: float
    # (x0, y0, x1, y1) in pixel coordinates of the image given to OCR,
    # or None when the engine returned no usable geometry.
    box: Optional[tuple]


_engine = None
_engine_lock = threading.Lock()

# PaddleOCR pipelines are not guaranteed to be thread-safe, and the upload
# route may run several uploads in worker threads.
_predict_lock = threading.Lock()


def is_ocr_enabled() -> bool:
    return OCR_ENABLED


def _build_engine():
    """Create the PaddleOCR pipeline. Called once, under a lock."""

    try:
        # Windows compatibility: load PyTorch before PaddlePaddle/PaddleOCR.
        # This preserves the import order verified in the DocuMind environment.
        import torch  # noqa: F401
        import paddle  # noqa: F401
        from paddleocr import PaddleOCR
    except Exception as error:
        logger.warning("PaddleOCR could not be imported: %s", error)
        raise OCRUnavailableError(
            "PaddleOCR is not installed or could not be loaded."
        ) from error

    base_arguments = {
        "text_detection_model_name": OCR_DETECTION_MODEL,
        "text_recognition_model_name": OCR_RECOGNITION_MODEL,
        # Page orientation / unwarping / text-line orientation models are
        # not needed for flat, upright pages and slow CPU inference down.
        # Keeping unwarping off also keeps the returned coordinates in the
        # coordinate system of the image that was passed in.
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "device": "cpu",
    }

    if OCR_DET_BOX_THRESH is not None:
        base_arguments["text_det_box_thresh"] = OCR_DET_BOX_THRESH

    if OCR_DET_UNCLIP_RATIO is not None:
        base_arguments["text_det_unclip_ratio"] = OCR_DET_UNCLIP_RATIO

    # PaddleOCR raises ValueError("Unknown argument: ...") for options it does
    # not know. If an installed version rejects one of the CPU tuning
    # options, fall back to the next, smaller set instead of failing.
    attempts = [
        {
            **base_arguments,
            "cpu_threads": OCR_CPU_THREADS,
            "enable_mkldnn": OCR_ENABLE_MKLDNN,
        },
        {**base_arguments, "cpu_threads": OCR_CPU_THREADS},
        base_arguments,
    ]

    last_error = None

    for arguments in attempts:
        try:
            return PaddleOCR(**arguments)
        except (TypeError, ValueError) as error:
            last_error = error
            logger.warning(
                "PaddleOCR rejected the options %s: %s",
                sorted(arguments),
                error
            )

    raise OCRUnavailableError(
        "The OCR engine could not be initialised."
    ) from last_error


def get_ocr_engine():
    """Return the shared PaddleOCR pipeline, creating it on first use."""

    global _engine

    if not OCR_ENABLED:
        raise OCRUnavailableError("OCR is disabled (OCR_ENABLED=false).")

    if _engine is not None:
        return _engine

    with _engine_lock:
        if _engine is None:
            try:
                logger.info(
                    "Loading OCR models (%s + %s) on CPU...",
                    OCR_DETECTION_MODEL,
                    OCR_RECOGNITION_MODEL
                )
                _engine = _build_engine()
                logger.info("OCR models loaded.")
            except OCRUnavailableError:
                raise
            except Exception as error:
                logger.exception("OCR initialisation failed")
                raise OCRUnavailableError(
                    "The OCR engine could not be initialised."
                ) from error

    return _engine


def crop_dark_borders(image: np.ndarray) -> tuple:
    """
    Remove solid black borders around a photographed or scanned page.

    Only rows and columns that are essentially pure dark (almost no bright
    pixels) are removed, so dark text areas are never cut off, and nothing is
    cropped unless it removes a meaningful part of the image.

    Returns (cropped_image, offset_x, offset_y) where the offsets give the
    position of the cropped image inside the original image in pixels.
    """

    height, width = image.shape[:2]

    if height < 64 or width < 64:
        return image, 0, 0

    # The green channel is a good, cheap brightness estimate.
    bright = image[:, :, 1] > 60

    row_has_content = bright.mean(axis=1) > 0.01

    if not row_has_content.any():
        return image, 0, 0

    content_rows = np.flatnonzero(row_has_content)
    top, bottom = int(content_rows[0]), int(content_rows[-1]) + 1

    col_has_content = bright[top:bottom].mean(axis=0) > 0.01

    if not col_has_content.any():
        return image, 0, 0

    content_cols = np.flatnonzero(col_has_content)
    left, right = int(content_cols[0]), int(content_cols[-1]) + 1

    margin = 8
    top = max(0, top - margin)
    left = max(0, left - margin)
    bottom = min(height, bottom + margin)
    right = min(width, right + margin)

    cropped_area = (bottom - top) * (right - left)

    # Ignore tiny crops, and never keep a degenerate sliver.
    if cropped_area > 0.95 * height * width or cropped_area < 0.10 * height * width:
        return image, 0, 0

    return np.ascontiguousarray(image[top:bottom, left:right]), left, top


def _to_list(value) -> list:
    if value is None:
        return []

    if isinstance(value, np.ndarray):
        return list(value)

    if isinstance(value, (str, bytes)):
        return []

    try:
        return list(value)
    except TypeError:
        return []


def _result_mapping(result) -> dict:
    """
    Return the dictionary holding rec_texts / rec_scores / rec_polys.

    A PaddleOCR 3.x result is a dict subclass whose keys (rec_texts,
    rec_scores, rec_polys, rec_boxes, dt_polys, ...) are directly accessible.
    Its ".json" property wraps the same fields in {"res": {...}}, which is
    also accepted here.
    """

    if isinstance(result, dict):
        inner = result.get("res")
        return inner if isinstance(inner, dict) else result

    as_json = getattr(result, "json", None)

    if isinstance(as_json, dict):
        inner = as_json.get("res")
        return inner if isinstance(inner, dict) else as_json

    return {}


def _box_from_polygon(polygon) -> Optional[tuple]:
    try:
        points = np.asarray(polygon, dtype=float).reshape(-1, 2)
    except (TypeError, ValueError):
        return None

    if points.size == 0 or not np.isfinite(points).all():
        return None

    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max())
    )


def _box_from_rectangle(rectangle) -> Optional[tuple]:
    try:
        values = np.asarray(rectangle, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None

    if values.size != 4 or not np.isfinite(values).all():
        return None

    return tuple(float(value) for value in values)


def _parse_result(result) -> list:
    """
    Convert one PaddleOCR result into OCRLine objects.

    rec_texts, rec_scores, rec_polys and rec_boxes describe the same lines
    in the same order (PaddleOCR applies its own score filter to all of them
    together). Geometry is therefore only trusted when its length matches the
    number of texts; otherwise the text is kept without geometry rather than
    being attached to the wrong box.
    """

    data = _result_mapping(result)

    texts = _to_list(data.get("rec_texts"))

    if not texts and data.get("error"):
        raise OCRInferenceError("OCR failed for this page.")

    scores = _to_list(data.get("rec_scores"))
    polygons = _to_list(data.get("rec_polys"))
    rectangles = _to_list(data.get("rec_boxes"))

    use_polygons = len(polygons) == len(texts)
    use_rectangles = len(rectangles) == len(texts)

    if texts and not (use_polygons or use_rectangles):
        logger.warning(
            "OCR returned %s texts but %s polygons and %s boxes; "
            "keeping the text without positions.",
            len(texts), len(polygons), len(rectangles)
        )

    if texts and len(scores) != len(texts):
        logger.warning(
            "OCR returned %s texts but %s scores; "
            "the confidence filter is not applied to this result.",
            len(texts), len(scores)
        )

    lines = []

    for index, text in enumerate(texts):
        text = str(text).strip()

        if not text:
            continue

        if len(scores) == len(texts):
            try:
                score = float(scores[index])
            except (TypeError, ValueError):
                score = 1.0
        else:
            score = 1.0

        box = None

        if use_polygons:
            box = _box_from_polygon(polygons[index])

        if box is None and use_rectangles:
            box = _box_from_rectangle(rectangles[index])

        lines.append(OCRLine(text=text, score=score, box=box))

    return lines


def recognize_image(image: np.ndarray, rescue_if_empty: bool = False) -> list:
    """
    Run OCR on one BGR uint8 image (height x width x 3).

    Returns a list of OCRLine. Lines without any letter or digit are dropped,
    and lines below OCR_MIN_CONFIDENCE are dropped (and reported in the log).

    rescue_if_empty: when every recognised line is below the threshold, keep
    them all instead of returning nothing. Used for pages that have no digital
    text at all, where weak text (for example faint handwriting) is better than
    an empty page.

    Raises OCRUnavailableError if the engine cannot be used and
    OCRInferenceError if recognition of this image fails.
    """

    engine = get_ocr_engine()

    try:
        with _predict_lock:
            # predict() returns a list in PaddleOCR 3.x, but consume it fully
            # inside the lock in case an iterator is returned.
            results = list(engine.predict(image))
    except Exception as error:
        logger.exception("OCR inference failed")
        raise OCRInferenceError("OCR failed for this page.") from error

    candidates = []

    for result in results:
        for line in _parse_result(result):
            if any(character.isalnum() for character in line.text):
                candidates.append(line)

    kept = [line for line in candidates if line.score >= OCR_MIN_CONFIDENCE]
    dropped = [line for line in candidates if line.score < OCR_MIN_CONFIDENCE]

    if dropped:
        logger.info(
            "OCR dropped %s of %s line(s) below the confidence threshold "
            "%.2f (highest dropped score %.2f). Lower OCR_MIN_CONFIDENCE if "
            "handwriting is being lost.",
            len(dropped),
            len(candidates),
            OCR_MIN_CONFIDENCE,
            max(line.score for line in dropped)
        )

    if not kept and candidates and rescue_if_empty:
        logger.warning(
            "All %s recognised line(s) were below the confidence threshold "
            "%.2f; keeping them because the page has no other text.",
            len(candidates),
            OCR_MIN_CONFIDENCE
        )
        return candidates

    return kept


def warm_up() -> None:
    """Load the models (downloading them on the first run)."""

    engine = get_ocr_engine()
    blank_image = np.full((96, 320, 3), 255, dtype=np.uint8)

    with _predict_lock:
        list(engine.predict(blank_image))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        warm_up()
        print("OCR models are ready.")
    except OCRUnavailableError as error:
        print(f"OCR is not available: {error}")
        raise SystemExit(1)
