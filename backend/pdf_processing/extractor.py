
import pymupdf


def extract_text_from_pdf(pdf_bytes: bytes) -> list:
    """
    Extract text from a PDF file provided as bytes.

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

        if pdf_document.is_encrypted:
            pdf_document.close()
            raise ValueError("Password-protected PDFs are not supported.")

        extracted_pages = []

        for page_number, page in enumerate(pdf_document, start=1):
            page_text = page.get_text("text").strip()

            if page_text:
                extracted_pages.append({
                    "page": page_number,
                    "text": page_text
                })

        pdf_document.close()

        if not extracted_pages:
            raise ValueError(
                "No readable text found. This PDF may contain scanned images."
            )

        return extracted_pages

    except ValueError:
        raise

    except Exception as error:
        raise ValueError(
            "Unable to read this PDF. Please upload a valid PDF file."
        ) from error
