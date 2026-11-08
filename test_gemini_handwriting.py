"""
Test Gemini Vision on one PDF page containing handwriting.

Run from the project root:
    python test_gemini_handwriting.py "eng2_p.pdf"
    python test_gemini_handwriting.py "eng2_p.pdf" --page 2
    python test_gemini_handwriting.py "eng2_p.pdf" --dpi 300

This is a standalone experiment. It does not modify the production PDF extractor.
"""

import argparse
import sys
from pathlib import Path

import pymupdf
from google import genai
from google.genai import types

# Make the project root importable when this script is run from another folder.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import GEMINI_API_KEY, GEMINI_MODEL


PROMPT = """
Transcribe the handwritten text visible in this PDF page image.

Rules:
1. Read the handwriting carefully and preserve the original wording.
2. Keep the original line and paragraph order as much as possible.
3. Do not summarize, explain, correct grammar, or add missing words.
4. Do not guess. If a word cannot be read confidently, write [unclear].
5. Return only the transcription, with no introduction or commentary.
"""


def main():
    parser = argparse.ArgumentParser(
        description="Test Gemini Vision handwriting transcription on a PDF page."
    )
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="1-based page number to test (default: 1)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=250,
        help="PDF rendering resolution (default: 250 DPI)",
    )
    parser.add_argument(
        "--model",
        default=GEMINI_MODEL,
        help="Gemini model name (defaults to GEMINI_MODEL in backend/config.py)",
    )
    args = parser.parse_args()

    if not GEMINI_API_KEY:
        raise SystemExit(
            "GEMINI_API_KEY is missing. Check the .env file in your project root."
        )

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.is_file():
        raise SystemExit(f"PDF file not found: {pdf_path}")

    if args.page < 1:
        raise SystemExit("--page must be 1 or greater.")
    if args.dpi < 72 or args.dpi > 400:
        raise SystemExit("--dpi must be between 72 and 400.")

    try:
        document = pymupdf.open(pdf_path)
    except Exception as exc:
        raise SystemExit(f"Could not open the PDF: {exc}") from exc

    try:
        if args.page > len(document):
            raise SystemExit(
                f"This PDF has {len(document)} page(s); page {args.page} does not exist."
            )

        page = document[args.page - 1]
        zoom = args.dpi / 72.0
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(zoom, zoom),
            colorspace=pymupdf.csRGB,
            alpha=False,
        )
        image_bytes = pixmap.tobytes("png")
    finally:
        document.close()

    print(f"PDF: {pdf_path.name}")
    print(f"Page: {args.page}")
    print(f"Rendered image: {pixmap.width} x {pixmap.height} pixels")
    print(f"Gemini model: {args.model}")
    print("Sending this page to Gemini Vision...")

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/png",
        )
        response = client.models.generate_content(
            model=args.model,
            contents=[PROMPT, image_part],
        )
    except Exception as exc:
        raise SystemExit(
            "\nGemini request failed. Check your API key, model name, internet "
            f"connection, and API quota.\nDetails: {exc}"
        ) from exc

    result = response.text
    if not result or not result.strip():
        raise SystemExit(
            "Gemini returned no text. Try another page or a different supported model."
        )

    print("\n" + "=" * 20 + " GEMINI TRANSCRIPTION " + "=" * 20)
    print(result.strip())
    print("=" * 62)


if __name__ == "__main__":
    main()
