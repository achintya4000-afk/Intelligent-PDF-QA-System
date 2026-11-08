
import torch
import paddle
import pymupdf
import tempfile
from pathlib import Path
from paddleocr import PaddleOCR

# Keep this import order for your Windows environment.

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "ocr_test_pdfs"
OUTPUT_DIR = BASE_DIR / "ocr_test_results"

OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in ocr_test_pdfs.")
        return

    print("PyTorch:", torch.__version__)
    print("PaddlePaddle:", paddle.__version__)
    print("Initializing OCR engine...")

    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="en_PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )

    for pdf_path in pdf_files:
        print("\n" + "=" * 65)
        print("PDF:", pdf_path.name)
        print("=" * 65)

        output_text = []

        try:
            document = pymupdf.open(pdf_path)

            if document.is_encrypted:
                print("Skipped: Password-protected PDF.")
                document.close()
                continue

            with tempfile.TemporaryDirectory() as temp_dir:
                for page_index, page in enumerate(document):
                    page_number = page_index + 1

                    print(f"\nProcessing page {page_number}...")

                    pixmap = page.get_pixmap(
                        dpi=200,
                        alpha=False
                    )

                    image_path = (
                        Path(temp_dir) / f"page_{page_number}.png"
                    )
                    pixmap.save(str(image_path))

                    results = ocr.predict(input=str(image_path))

                    page_lines = []

                    for result in results:
                        data = result.json
                        if callable(data):
                            data = data()

                        if not isinstance(data, dict):
                            raise ValueError(
                                "Unexpected PaddleOCR result format."
                            )

                        data = data.get("res", data)

                        texts = data.get("rec_texts", [])
                        scores = data.get("rec_scores", [])

                        if len(texts) != len(scores):
                            raise ValueError(
                                "OCR text and confidence counts differ."
                            )

                        for text, score in zip(texts, scores):
                            text = str(text).strip()

                            if text:
                                page_lines.append(
                                    f"[{float(score):.3f}] {text}"
                                )

                    output_text.append(
                        f"\n--- PAGE {page_number} ---\n"
                    )
                    output_text.extend(page_lines)

                    print("Recognized lines:", len(page_lines))

                    for line in page_lines:
                        print(line)

            document.close()

            output_file = (
                OUTPUT_DIR / f"{pdf_path.stem}_ocr.txt"
            )

            output_file.write_text(
                "\n".join(output_text),
                encoding="utf-8"
            )

            print("\nSaved OCR output:", output_file)

        except Exception as error:
            print(f"ERROR processing {pdf_path.name}: {error}")

    print("\nOCR testing completed.")


if __name__ == "__main__":
    main()
