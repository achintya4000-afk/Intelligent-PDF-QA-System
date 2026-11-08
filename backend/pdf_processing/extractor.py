import logging

import numpy as np
import pymupdf

from backend.pdf_processing import ocr_engine
from backend.pdf_processing.ocr_engine import (
    OCRInferenceError,
    OCRUnavailableError
)

logger = logging.getLogger(__name__)

# Pages with fewer non-space characters than this are treated as having
# no usable selectable text.
MIN_DIGITAL_CHARACTERS = 20

# On a page that already has digital text, images are OCR'd region by region.
# Once images cover at least this share of the page, the whole page is OCR'd.
FULL_PAGE_IMAGE_COVERAGE = 0.5

# An image narrower or shorter than this (in PDF points, 72 per inch) is
# treated as an icon, rule or decoration and is not OCR'd on a text page.
MIN_IMAGE_SIDE_POINTS = 36.0

# More image regions than this on one page: OCR the whole page instead.
MAX_IMAGE_REGIONS_PER_PAGE = 6

# Padding (in points) added around an image region before rendering it.
REGION_PADDING_POINTS = 4.0

# Annotation types whose appearance is not part of the page's text:
# free text, stamp and ink (hand-drawn) annotations.
_HANDWRITING_ANNOTATION_TYPES = {
    getattr(pymupdf, "PDF_ANNOT_FREE_TEXT", 2),
    getattr(pymupdf, "PDF_ANNOT_STAMP", 13),
    getattr(pymupdf, "PDF_ANNOT_INK", 15),
}

# An OCR line is a duplicate of digital text when at least this share of its
# box is covered by digital words, or when only a small share is covered but
# its text is already present in the digital text.
DUPLICATE_OVERLAP_RATIO = 0.5
DUPLICATE_PARTIAL_OVERLAP_RATIO = 0.15


class _ExtractionState:
    """Collects what happened while a document is processed."""

    def __init__(self):
        self.ocr_attempted_pages = 0
        self.ocr_failed_pages = 0
        self.ocr_unavailable = False


def extract_text_from_pdf(pdf_bytes: bytes) -> list:
    """
    Extract text from a PDF file provided as bytes.

    Digital text is read with PyMuPDF. Pages that are scanned, handwritten or
    mixed (selectable text plus scanned images or annotations) are also
    recognised locally with PaddleOCR, and both results are combined without
    duplicates.

    Returns a list containing the text and page number
    for each page.
    """

    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    try:
        pdf_document = pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf"
        )
    except Exception as error:
        raise ValueError(
            "Unable to read this PDF. Please upload a valid PDF file."
        ) from error

    state = _ExtractionState()

    try:
        if pdf_document.is_encrypted:
            raise ValueError("Password-protected PDFs are not supported.")

        if len(pdf_document) == 0:
            raise ValueError("This PDF has no pages.")

        extracted_pages = []

        # Pages are processed one at a time; each rendered image is released
        # before the next page, so memory use does not grow with page count.
        for page_number, page in enumerate(pdf_document, start=1):
            try:
                page_text = _extract_page_text(page, page_number, state)
            except Exception:
                logger.exception(
                    "Unexpected error while extracting page %s", page_number
                )
                continue

            if page_text:
                extracted_pages.append({
                    "page": page_number,
                    "text": page_text
                })

        if state.ocr_attempted_pages:
            logger.info(
                "OCR processed %s page(s); %s failed.",
                state.ocr_attempted_pages,
                state.ocr_failed_pages
            )

        if not extracted_pages:
            raise ValueError(_no_text_message(state))

        return extracted_pages

    except ValueError:
        raise

    except Exception as error:
        raise ValueError(
            "Unable to read this PDF. Please upload a valid PDF file."
        ) from error

    finally:
        pdf_document.close()


def _no_text_message(state: _ExtractionState) -> str:
    if state.ocr_unavailable:
        return (
            "This PDF appears to be scanned, but local OCR is not available. "
            "Install the OCR dependencies from requirements.txt and "
            "restart the backend."
        )

    if state.ocr_attempted_pages:
        return (
            "No readable text could be recognized in this PDF. "
            "The scan may be blank, too faint or too low in quality."
        )

    return "No readable text found. This PDF may contain scanned images."


# ---------------------------------------------------------------------------
# Per-page extraction
# ---------------------------------------------------------------------------

def _extract_page_text(page, page_number: int, state: _ExtractionState) -> str:
    try:
        digital_text = page.get_text("text").strip()
    except Exception:
        logger.exception("Digital text extraction failed on page %s", page_number)
        digital_text = ""

    if state.ocr_unavailable or not ocr_engine.is_ocr_enabled():
        return digital_text

    plan = _plan_ocr(page, digital_text)

    if not plan:
        return digital_text

    state.ocr_attempted_pages += 1

    # Weak text is better than an empty page only when the page has no
    # digital text of its own.
    rescue_if_empty = not digital_text

    try:
        ocr_lines = _ocr_regions(page, plan, rescue_if_empty)
    except OCRUnavailableError as error:
        # The engine cannot start; do not retry it for every remaining page.
        logger.warning("OCR unavailable: %s", error)
        state.ocr_attempted_pages -= 1
        state.ocr_unavailable = True
        return digital_text
    except OCRInferenceError:
        logger.warning("OCR failed on page %s", page_number)
        state.ocr_failed_pages += 1
        return digital_text
    except Exception:
        logger.exception("OCR failed on page %s", page_number)
        state.ocr_failed_pages += 1
        return digital_text

    return _merge_page_text(page, digital_text, ocr_lines)


def _plan_ocr(page, digital_text: str):
    """
    Decide what to OCR on a page.

    Returns None (no OCR) or a list of regions. Each region is a
    pymupdf.Rect, or None meaning "the whole page".

    - OCR_MODE=always: every page is OCR'd completely.
    - No usable digital text: the whole page is OCR'd when it contains any
      image, annotation or drawing (scanned, photographed or handwritten
      pages). A truly empty page is skipped.
    - Digital text present (possible mixed content): OCR is not skipped just
      because text was found. Hand-drawn / free-text / stamp annotations
      trigger a full-page OCR; otherwise every image that is large enough to
      hold writing is OCR'd on its own, or the whole page when images cover
      most of it. Ordinary digital pages (no such images or annotations) are
      not OCR'd.
    """

    if ocr_engine.OCR_MODE == "always":
        return [None]

    character_count = len("".join(digital_text.split()))
    has_digital_text = character_count >= MIN_DIGITAL_CHARACTERS

    if not has_digital_text:
        if (
            _has_any_image(page)
            or _has_handwriting_annotation(page)
            or _has_drawings(page)
        ):
            return [None]

        return None

    if _has_handwriting_annotation(page):
        return [None]

    regions, coverage = _image_regions(page)

    if not regions:
        return None

    if (
        coverage >= FULL_PAGE_IMAGE_COVERAGE
        or len(regions) > MAX_IMAGE_REGIONS_PER_PAGE
    ):
        return [None]

    return regions


def _has_any_image(page) -> bool:
    try:
        return len(page.get_image_info()) > 0
    except Exception:
        return False


def _has_drawings(page) -> bool:
    try:
        return bool(page.get_drawings())
    except Exception:
        return False


def _has_handwriting_annotation(page) -> bool:
    """True when the page has ink, free-text or stamp annotations."""

    try:
        annotations = page.annots()

        if annotations is None:
            return False

        for annotation in annotations:
            annotation_type = annotation.type

            if (
                annotation_type
                and annotation_type[0] in _HANDWRITING_ANNOTATION_TYPES
            ):
                return True
    except Exception:
        return False

    return False


def _merge_rectangles(rectangles: list) -> list:
    """Merge rectangles that overlap or touch into their union."""

    merged = [list(rectangle) for rectangle in rectangles]
    changed = True

    while changed and len(merged) > 1:
        changed = False

        for first_index in range(len(merged)):
            for second_index in range(first_index + 1, len(merged)):
                first = merged[first_index]
                second = merged[second_index]

                if (
                    first[0] <= second[2] and second[0] <= first[2]
                    and first[1] <= second[3] and second[1] <= first[3]
                ):
                    merged[first_index] = [
                        min(first[0], second[0]),
                        min(first[1], second[1]),
                        max(first[2], second[2]),
                        max(first[3], second[3])
                    ]
                    del merged[second_index]
                    changed = True
                    break

            if changed:
                break

    return merged


def _image_regions(page) -> tuple:
    """
    Images on the page that are large enough to contain writing.

    Returns (regions, coverage): padded pymupdf.Rect objects clipped to the
    page, and the share (0..1) of the page covered by them. Images that are
    tiny (icons, lines, decorations) are ignored.
    """

    try:
        image_infos = page.get_image_info()
    except Exception:
        return [], 0.0

    page_rect = page.rect
    page_area = page_rect.width * page_rect.height

    if page_area <= 0:
        return [], 0.0

    rectangles = []

    for info in image_infos:
        bbox = info.get("bbox")

        if not bbox or len(bbox) != 4:
            continue

        x0 = max(bbox[0], page_rect.x0)
        y0 = max(bbox[1], page_rect.y0)
        x1 = min(bbox[2], page_rect.x1)
        y1 = min(bbox[3], page_rect.y1)

        width = x1 - x0
        height = y1 - y0

        if width < MIN_IMAGE_SIDE_POINTS or height < MIN_IMAGE_SIDE_POINTS:
            continue

        if (width * height) / page_area < ocr_engine.OCR_MIN_IMAGE_AREA_RATIO:
            continue

        rectangles.append([x0, y0, x1, y1])

    if not rectangles:
        return [], 0.0

    merged = _merge_rectangles(rectangles)

    covered_area = sum((r[2] - r[0]) * (r[3] - r[1]) for r in merged)
    coverage = min(covered_area / page_area, 1.0)

    regions = []

    for x0, y0, x1, y1 in merged:
        regions.append(pymupdf.Rect(
            max(page_rect.x0, x0 - REGION_PADDING_POINTS),
            max(page_rect.y0, y0 - REGION_PADDING_POINTS),
            min(page_rect.x1, x1 + REGION_PADDING_POINTS),
            min(page_rect.y1, y1 + REGION_PADDING_POINTS)
        ))

    return regions, coverage


# ---------------------------------------------------------------------------
# Rendering and OCR
# ---------------------------------------------------------------------------

def _ocr_regions(page, regions: list, rescue_if_empty: bool) -> list:
    """
    OCR the given regions (None = whole page) of one page.

    A failure in one region does not discard the others. If every region
    fails, OCRInferenceError is raised.
    """

    lines = []
    failures = 0

    for region in regions:
        try:
            lines.extend(_ocr_region(page, region, rescue_if_empty))
        except OCRUnavailableError:
            raise
        except Exception:
            failures += 1
            logger.exception("OCR failed for one region of a page")

    if failures and failures == len(regions):
        raise OCRInferenceError("OCR failed for this page.")

    return lines


def _ocr_region(page, region, rescue_if_empty: bool) -> list:
    """
    Render one region (or the whole page), run OCR and return lines with
    boxes in PDF points.

    Each returned item is (x0, y0, x1, y1, text), or (None, None, None, None,
    text) when the OCR engine gave no usable geometry.
    """

    target_rect = region if region is not None else page.rect

    if target_rect.width <= 0 or target_rect.height <= 0:
        return []

    zoom = ocr_engine.OCR_RENDER_DPI / 72.0
    longest_side = max(target_rect.width, target_rect.height) * zoom

    if longest_side > ocr_engine.OCR_MAX_IMAGE_SIDE:
        zoom *= ocr_engine.OCR_MAX_IMAGE_SIDE / longest_side

    # Annotations are rendered too (get_pixmap draws them by default), so
    # hand-drawn and free-text annotations become visible to OCR.
    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(zoom, zoom),
        clip=region,
        colorspace=pymupdf.csRGB,
        alpha=False
    )

    try:
        pixel_width = pixmap.width
        pixel_height = pixmap.height
        origin_x = pixmap.x
        origin_y = pixmap.y

        rgb = np.frombuffer(pixmap.samples, dtype=np.uint8)
        rgb = rgb.reshape(pixel_height, pixmap.stride)
        rgb = rgb[:, :pixel_width * 3].reshape(pixel_height, pixel_width, 3)

        # PaddleOCR expects BGR images (OpenCV convention).
        image = np.ascontiguousarray(rgb[:, :, ::-1])
    finally:
        pixmap = None

    image, offset_x, offset_y = ocr_engine.crop_dark_borders(image)

    # PaddleOCR remains the primary local OCR engine for printed text.
    paddle_lines = ocr_engine.recognize_image(
        image,
        rescue_if_empty=rescue_if_empty
    )

    # Gemini is an optional handwriting recogniser. It returns the locations
    # and transcription of handwritten regions only. If the API is unavailable
    # or returns unusable coordinates, PaddleOCR output is retained unchanged.
    try:
        from backend.pdf_processing.handwriting_engine import (
            recognize_handwriting_regions,
            remove_printed_lines_overlapping_handwriting
        )

        handwriting_lines = recognize_handwriting_regions(image)
        lines = remove_printed_lines_overlapping_handwriting(
            paddle_lines,
            handwriting_lines
        )
        lines.extend(handwriting_lines)
    except Exception:
        logger.exception(
            "Hybrid handwriting OCR failed; falling back to PaddleOCR only."
        )
        lines = paddle_lines

    del image

    recognised = []

    for line in lines:
        if line.box is None:
            recognised.append((None, None, None, None, line.text))
            continue

        x0, y0, x1, y1 = line.box

        # Pixel position in the rendered image -> position on the page in
        # PDF points (pixmap origin and zoom undo the render transform).
        recognised.append((
            (origin_x + x0 + offset_x) / zoom,
            (origin_y + y0 + offset_y) / zoom,
            (origin_x + x1 + offset_x) / zoom,
            (origin_y + y1 + offset_y) / zoom,
            line.text
        ))

    return recognised


# ---------------------------------------------------------------------------
# Combining digital text and OCR text
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return "".join(character for character in text.lower() if character.isalnum())


def _digital_word_boxes(page):
    """Boxes of digital words as an (N, 4) array, or None if unusable."""

    # Coordinates of rotated pages are not guaranteed to match the rendered
    # image, so position-based comparison is skipped for them.
    if page.rotation != 0:
        return None

    try:
        words = page.get_text("words")
    except Exception:
        return None

    if not words:
        return None

    return np.array([word[:4] for word in words], dtype=float)


def _overlap_ratio(box: tuple, word_boxes) -> float:
    """Share of the OCR box area that is covered by digital words."""

    x0, y0, x1, y1 = box
    area = (x1 - x0) * (y1 - y0)

    if area <= 0:
        return 0.0

    intersect_width = np.clip(
        np.minimum(word_boxes[:, 2], x1) - np.maximum(word_boxes[:, 0], x0),
        0,
        None
    )
    intersect_height = np.clip(
        np.minimum(word_boxes[:, 3], y1) - np.maximum(word_boxes[:, 1], y0),
        0,
        None
    )

    return float((intersect_width * intersect_height).sum() / area)


def _is_duplicate(line: tuple, digital_normalized: str, word_boxes) -> bool:
    """
    True when an OCR line only repeats digital text.

    Handwritten additions are kept: a line is dropped only when it sits on
    top of digital text (position) or sits near it and repeats it (text).
    """

    x0, y0, x1, y1, text = line
    normalized = _normalize(text)

    if not normalized:
        return True

    contained = len(normalized) >= 6 and normalized in digital_normalized

    if x0 is not None and word_boxes is not None:
        overlap = _overlap_ratio((x0, y0, x1, y1), word_boxes)

        if overlap >= DUPLICATE_OVERLAP_RATIO:
            return True

        return overlap >= DUPLICATE_PARTIAL_OVERLAP_RATIO and contained

    return len(normalized) >= 8 and contained


def _merge_page_text(page, digital_text: str, ocr_lines: list) -> str:
    if not ocr_lines:
        return digital_text

    if not digital_text:
        # Without geometry for every line, keep the engine's own order.
        if any(line[0] is None for line in ocr_lines):
            return "\n".join(line[4] for line in ocr_lines).strip()

        return _ordered_text(_items_from_ocr(ocr_lines), page.rect.width)

    digital_normalized = _normalize(digital_text)
    word_boxes = _digital_word_boxes(page)

    new_lines = [
        line for line in ocr_lines
        if not _is_duplicate(line, digital_normalized, word_boxes)
    ]

    if not new_lines:
        return digital_text

    # Without geometry the new lines cannot be placed in reading order,
    # so they are appended after the digital text.
    if any(line[0] is None for line in new_lines):
        extra_text = "\n".join(line[4] for line in new_lines)
        return f"{digital_text}\n{extra_text}".strip()

    items = _digital_block_items(page)
    items.extend(_items_from_ocr(new_lines))

    if not items:
        extra_text = "\n".join(line[4] for line in new_lines)
        return f"{digital_text}\n{extra_text}".strip()

    return _ordered_text(items, page.rect.width)


def _items_from_ocr(ocr_lines: list) -> list:
    items = []

    for x0, y0, x1, y1, text in ocr_lines:
        if x0 is None:
            continue

        items.append((x0, y0, x1, y1, text))

    return items


def _digital_block_items(page) -> list:
    try:
        blocks = page.get_text("blocks")
    except Exception:
        return []

    items = []

    for block in blocks:
        # block = (x0, y0, x1, y1, text, block_no, block_type); 0 = text
        if len(block) < 7 or block[6] != 0:
            continue

        text = str(block[4]).strip()

        if text:
            items.append((block[0], block[1], block[2], block[3], text))

    return items


def _ordered_text(items: list, page_width: float) -> str:
    ordered = _reading_order(items, page_width)

    return "\n".join(item[4] for item in ordered).strip()


def _split_groups(group: list, axis: int, min_gap: float, shrink: float) -> list:
    """
    Split items into runs separated by an empty gap along one axis.

    axis 0 splits left to right (columns), axis 1 top to bottom (bands).
    `shrink` trims each item's interval by that share of its size so that
    slightly overlapping line boxes still count as separate rows.
    """

    start_index, end_index = (0, 2) if axis == 0 else (1, 3)

    def interval(item):
        start, end = item[start_index], item[end_index]
        trim = (end - start) * shrink
        return start + trim, end - trim

    ordered = sorted(group, key=lambda item: interval(item)[0])

    runs = [[ordered[0]]]
    current_end = interval(ordered[0])[1]

    for item in ordered[1:]:
        start, end = interval(item)

        if start - current_end > min_gap:
            runs.append([item])
            current_end = end
        else:
            runs[-1].append(item)
            current_end = max(current_end, end)

    return runs


def _reading_order(items: list, page_width: float) -> list:
    """
    Order text items in reading order with a simple recursive XY-cut:
    split into horizontal bands, split bands into columns, and read each
    column top to bottom. Falls back to top-to-bottom, left-to-right.
    """

    column_gap = max(page_width * 0.025, 8.0)
    result = []

    def top_to_bottom(group):
        return sorted(
            group,
            key=lambda item: ((item[1] + item[3]) / 2.0, item[0])
        )

    def process(group, depth):
        if len(group) <= 1 or depth > 20:
            result.extend(top_to_bottom(group))
            return

        bands = _split_groups(group, axis=1, min_gap=0.0, shrink=0.15)

        if len(bands) > 1:
            for band in bands:
                process(band, depth + 1)
            return

        columns = _split_groups(group, axis=0, min_gap=column_gap, shrink=0.0)

        if len(columns) > 1:
            for column in columns:
                process(column, depth + 1)
            return

        result.extend(top_to_bottom(group))

    process(list(items), 0)

    return result
