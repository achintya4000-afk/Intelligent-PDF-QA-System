"""
Standalone TrOCR handwriting test for DocuMind.

Purpose:
- Render PDF pages with PyMuPDF.
- Use PaddleOCR's text detector to locate text-line regions.
- Use Microsoft's TrOCR handwritten model to recognize each cropped line.
- Save page images, line crops, and recognized text for inspection.

This is a TEST SCRIPT. It does not modify DocuMind's extractor, routes, or OCR engine.
Run from the project root:
    python test_trocr_pdf.py "ocr_test_pdfs/eng2_p.pdf"

Important:
TrOCR is a line-level recognizer. It should not be given an entire page as if it
were one text line. PaddleOCR is used here only to propose line-region boxes.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import pymupdf
from PIL import Image, ImageOps

# Windows workaround already verified in this DocuMind environment:
# import PyTorch before PaddlePaddle/PaddleOCR.
import torch  # noqa: F401
import paddle  # noqa: F401
from paddleocr import PaddleOCR

from transformers import TrOCRProcessor, VisionEncoderDecoderModel


LOGGER = logging.getLogger("trocr_test")
MODEL_NAME = "microsoft/trocr-base-handwritten"
RENDER_DPI = 250
MAX_LINE_CROP_WIDTH = 1800
MAX_LINE_CROP_HEIGHT = 500
CROP_PADDING = 12
MAX_NEW_TOKENS = 128


def load_trocr():
    """Load TrOCR once on CPU."""
    LOGGER.info("Loading TrOCR model: %s", MODEL_NAME)
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    model.to("cpu")
    model.eval()
    LOGGER.info("TrOCR is ready on CPU.")
    return processor, model


def load_text_detector():
    """Load the same lightweight English PaddleOCR detector used by DocuMind."""
    LOGGER.info("Loading PaddleOCR text detector on CPU...")
    return PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="en_PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def get_result_dict(result):
    """Normalize PaddleOCR's result object to a dictionary."""
    data = getattr(result, "json", None)
    if callable(data):
        data = data()
    if data is None:
        raise RuntimeError("PaddleOCR returned a result without JSON data.")

    if isinstance(data, str):
        data = json.loads(data)

    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected PaddleOCR result type: {type(data)!r}")

    # PaddleOCR 3.x may wrap prediction fields inside the 'res' object.
    if isinstance(data.get("res"), dict):
        data = data["res"]

    return data


def polygon_to_box(polygon, image_width, image_height):
    """Convert a PaddleOCR quadrilateral/polygon to a clipped rectangle."""
    points = []
    for point in polygon:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        points.append((float(point[0]), float(point[1])))

    if len(points) < 4:
        return None

    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]

    left = max(0, int(min(x_values)) - CROP_PADDING)
    top = max(0, int(min(y_values)) - CROP_PADDING)
    right = min(image_width, int(max(x_values)) + CROP_PADDING)
    bottom = min(image_height, int(max(y_values)) + CROP_PADDING)

    if right - left < 8 or bottom - top < 8:
        return None

    return left, top, right, bottom


def detect_line_boxes(detector, image_path, image_size):
    """Get text-region boxes from PaddleOCR's detection polygons."""
    results = detector.predict(str(image_path))
    if not results:
        return []

    data = get_result_dict(results[0])

    polygons = data.get("dt_polys")
    if polygons is None:
        polygons = data.get("rec_polys")

    if polygons is None:
        available = ", ".join(sorted(data.keys()))
        raise RuntimeError(
            "Could not find text detection polygons in PaddleOCR output. "
            f"Available fields: {available}"
        )

    width, height = image_size
    boxes = []

    for polygon in polygons:
        box = polygon_to_box(polygon, width, height)
        if box is not None:
            boxes.append(box)

    # Reading order: top-to-bottom, then left-to-right.
    boxes.sort(key=lambda box: (box[1], box[0]))
    return boxes


def resize_crop_for_trocr(crop: Image.Image) -> Image.Image:
    """Keep crops within a practical size while preserving aspect ratio."""
    crop = crop.convert("RGB")
    width, height = crop.size

    scale = min(
        1.0,
        MAX_LINE_CROP_WIDTH / max(width, 1),
        MAX_LINE_CROP_HEIGHT / max(height, 1),
    )

    if scale < 1.0:
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        crop = crop.resize(new_size, Image.Resampling.LANCZOS)

    # A white border helps prevent characters touching the crop boundary.
    return ImageOps.expand(crop, border=8, fill="white")


def recognize_crop(processor, model, crop: Image.Image) -> str:
    """Recognize one text-line crop using TrOCR on CPU."""
    pixel_values = processor(images=crop, return_tensors="pt").pixel_values

    with torch.inference_mode():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=1,
        )

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    return re.sub(r"\s+", " ", text).strip()


def render_pdf_pages(pdf_path: Path, output_dir: Path):
    """Render pages to PNG and yield (1-based page number, image path)."""
    document = pymupdf.open(pdf_path)

    if document.is_encrypted:
        document.close()
        raise ValueError("Password-protected PDFs are not supported.")

    if len(document) == 0:
        document.close()
        raise ValueError("The PDF contains no pages.")

    try:
        scale = RENDER_DPI / 72.0
        matrix = pymupdf.Matrix(scale, scale)

        for page_number, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(
                matrix=matrix,
                colorspace=pymupdf.csRGB,
                alpha=False,
            )
            image_path = output_dir / f"page_{page_number:03d}.png"
            pixmap.save(image_path)
            yield page_number, image_path
    finally:
        document.close()


def process_pdf(pdf_path: Path, output_dir: Path):
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = output_dir / "line_crops"
    crops_dir.mkdir(exist_ok=True)

    processor, model = load_trocr()
    detector = load_text_detector()

    all_pages = []

    for page_number, page_image_path in render_pdf_pages(pdf_path, output_dir):
        LOGGER.info("Detecting text lines on page %s...", page_number)

        with Image.open(page_image_path) as source:
            page_image = source.convert("RGB")

        boxes = detect_line_boxes(
            detector,
            page_image_path,
            page_image.size,
        )

        LOGGER.info("Detected %s candidate text line(s).", len(boxes))

        page_lines = []

        for line_number, box in enumerate(boxes, start=1):
            crop = page_image.crop(box)
            crop = resize_crop_for_trocr(crop)

            crop_path = crops_dir / (
                f"page_{page_number:03d}_line_{line_number:03d}.png"
            )
            crop.save(crop_path)

            try:
                recognized_text = recognize_crop(processor, model, crop)
            except Exception:
                LOGGER.exception(
                    "TrOCR failed on page %s, line %s",
                    page_number,
                    line_number,
                )
                recognized_text = ""

            if recognized_text:
                page_lines.append(recognized_text)

            print(
                f"[Page {page_number}, line {line_number}] "
                f"{recognized_text or '[NO TEXT RECOGNIZED]'}"
            )

        all_pages.append(
            {
                "page": page_number,
                "text": "\n".join(page_lines),
                "line_count": len(boxes),
                "recognized_line_count": sum(bool(line) for line in page_lines),
            }
        )

        del page_image

    result_path = output_dir / "trocr_output.json"
    result_path.write_text(
        json.dumps(all_pages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    text_path = output_dir / "trocr_output.txt"
    text_content = "\n\n".join(
        f"--- PAGE {item['page']} ---\n{item['text']}"
        for item in all_pages
    )
    text_path.write_text(text_content, encoding="utf-8")

    print("\n" + "=" * 60)
    print("TrOCR TEST COMPLETED")
    print(f"Recognized text: {text_path.resolve()}")
    print(f"JSON results:    {result_path.resolve()}")
    print(f"Line crops:      {crops_dir.resolve()}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Test TrOCR handwritten text recognition on a PDF."
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        default="ocr_test_pdfs/eng2_p.pdf",
        help="Path to the PDF to test.",
    )
    parser.add_argument(
        "--output",
        default="trocr_test_results",
        help="Directory for rendered pages, line crops, and OCR output.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    try:
        process_pdf(Path(args.pdf), Path(args.output))
    except Exception as error:
        LOGGER.exception("TrOCR test failed: %s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
