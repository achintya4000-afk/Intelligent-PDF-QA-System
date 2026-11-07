
import pymupdf

from backend.pdf_processing.handwriting_ocr import GoogleVisionOCR


def extract_text_from_pdf(pdf_bytes: bytes) -> list:
    """
    Extract printed and handwritten text from a PDF.

    PyMuPDF extracts selectable text.
    Google Cloud Vision analyses each rendered page to recognize
    both printed and handwritten content.

    Returns a list containing the text and original page number.
    """

    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    extracted_pages = []

    try:
        with pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf"
        ) as pdf_document:

            if pdf_document.is_encrypted:
                raise ValueError(
                    "Password-protected PDFs are not supported."
                )

            # Initialize Google Vision once per PDF, not once per page.
            ocr_client = GoogleVisionOCR()

            for page_number, page in enumerate(
                pdf_document,
                start=1
            ):
                # Extract selectable digital text.
                page_text = page.get_text("text").strip()

                # Render the complete page for OCR.
                page_image = page.get_pixmap(
                    dpi=200,
                    alpha=False
                )

                image_bytes = page_image.tobytes("png")

                # Recognize printed and handwritten text.
                ocr_text = ocr_client.extract_text(image_bytes)

                # Prefer full-page OCR to avoid duplicating
                # printed text extracted by PyMuPDF.
                final_text = ocr_text if ocr_text else page_text

                if final_text:
                    extracted_pages.append({
                        "page": page_number,
                        "text": final_text
                    })

        if not extracted_pages:
            raise ValueError(
                "No readable text found in this PDF."
            )

        return extracted_pages

    except (ValueError, RuntimeError):
        raise

    except Exception as error:
        raise ValueError(
            "Unable to process this PDF. "
            "Please upload a valid PDF file."
        ) from error
