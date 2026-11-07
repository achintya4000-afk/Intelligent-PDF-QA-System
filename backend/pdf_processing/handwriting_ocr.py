
"""
Google Cloud Vision OCR helper for DocuMind.

This module recognizes printed and handwritten text from a
single rendered PDF page image.

Authentication uses Google Application Default Credentials.
"""

from google.cloud import vision


class GoogleVisionOCR:
    """Recognize printed and handwritten text using Google Vision."""

    def __init__(self, timeout: int = 60):
        if (
            not isinstance(timeout, int)
            or isinstance(timeout, bool)
            or timeout <= 0
        ):
            raise ValueError("OCR timeout must be a positive integer.")

        try:
            self.client = vision.ImageAnnotatorClient()
        except Exception as error:
            raise RuntimeError(
                "Unable to initialize Google Cloud Vision. "
                "Check your Google Cloud credentials and API setup."
            ) from error

        self.timeout = timeout

    def extract_text(self, image_bytes: bytes) -> str:
        """
        Extract printed and handwritten text from page image bytes.

        Returns the recognized text as a string.
        Returns an empty string if no text is detected.
        """

        if not isinstance(image_bytes, bytes) or not image_bytes:
            raise ValueError("A valid, non-empty page image is required.")

        image = vision.Image(content=image_bytes)

        try:
            response = self.client.document_text_detection(
                image=image,
                timeout=self.timeout
            )

        except Exception as error:
            raise RuntimeError(
                "Google Cloud Vision OCR request failed."
            ) from error

        if response.error.message:
            raise RuntimeError(
                f"Google Cloud Vision error: {response.error.message}"
            )

        if not response.full_text_annotation:
            return ""

        return response.full_text_annotation.text.strip()
