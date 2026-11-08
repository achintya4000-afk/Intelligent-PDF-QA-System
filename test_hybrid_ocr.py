"""Run DocuMind's PDF extractor directly to inspect hybrid OCR output.

Usage from the project root:
    python test_hybrid_ocr.py "a01-003u.pdf"
    python test_hybrid_ocr.py "a01-072u.pdf"
"""

import sys
from pathlib import Path

from backend.pdf_processing.extractor import extract_text_from_pdf


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python test_hybrid_ocr.py "document.pdf"')

    pdf_path = Path(sys.argv[1]).expanduser()
    if not pdf_path.is_file():
        raise SystemExit(f"PDF file not found: {pdf_path.resolve()}")

    pages = extract_text_from_pdf(pdf_path.read_bytes())

    for item in pages:
        print(f"\n========== PAGE {item['page']} ==========\n")
        print(item["text"])


if __name__ == "__main__":
    main()
