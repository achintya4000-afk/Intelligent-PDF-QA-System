import json
import shutil
from pathlib import Path

import pymupdf

from backend.pdf_processing.extractor import extract_text_from_pdf
from backend.pdf_processing.chunker import chunk_text
from backend.pdf_processing.metadata import create_document_metadata
from backend.storage.document_store import (
    save_document,
    get_document,
    delete_document as delete_document_record
)


# Project directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Storage directories
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
INDEX_DIR = BASE_DIR / "data" / "indexes"

# Maximum PDF size: 20 MB
MAX_FILE_SIZE = 20 * 1024 * 1024


def upload_document(
    filename: str,
    pdf_bytes: bytes,
    chunk_size: int = 1000
) -> dict:
    """
    Process and store an uploaded PDF document.

    Steps:
    1. Validate the uploaded file.
    2. Extract text from the PDF.
    3. Divide extracted text into chunks.
    4. Create document metadata.
    5. Save the PDF and chunks.
    6. Save metadata in the document store.

    Returns the document metadata.
    """

    # Validate filename
    if not filename or not filename.strip():
        raise ValueError("Filename is required.")

    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")

    # Validate file content
    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    if len(pdf_bytes) > MAX_FILE_SIZE:
        raise ValueError("PDF size must not exceed 20 MB.")

    # Extract text from the PDF
    extracted_pages = extract_text_from_pdf(pdf_bytes)

    # Divide extracted text into chunks
    chunks = chunk_text(
        extracted_pages,
        chunk_size=chunk_size
    )

    if not chunks:
        raise ValueError("No readable text chunks were generated.")

    # Get PDF page count
    pdf_document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )
    total_pages = len(pdf_document)
    pdf_document.close()

    # Create document metadata
    metadata = create_document_metadata(
        filename=filename,
        file_size=len(pdf_bytes),
        total_pages=total_pages,
        total_chunks=len(chunks)
    )

    document_id = metadata["id"]

    # Create storage paths using the generated document ID
    pdf_path = UPLOAD_DIR / f"{document_id}.pdf"
    document_index_dir = INDEX_DIR / document_id
    chunks_path = document_index_dir / "chunks.json"

    try:
        # Create storage directories
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        document_index_dir.mkdir(parents=True, exist_ok=True)

        # Save original PDF
        pdf_path.write_bytes(pdf_bytes)

        # Save extracted chunks
        with chunks_path.open("w", encoding="utf-8") as file:
            json.dump(chunks, file, indent=4, ensure_ascii=False)

        # Store relative paths in metadata
        metadata["file_path"] = pdf_path.relative_to(BASE_DIR).as_posix()
        metadata["chunks_path"] = chunks_path.relative_to(BASE_DIR).as_posix()

        # Save metadata in JSON document store
        save_document(metadata)

    except Exception:
        # Remove partially saved files if processing/storage fails
        if pdf_path.exists():
            pdf_path.unlink()

        if document_index_dir.exists():
            shutil.rmtree(document_index_dir)

        raise

    return metadata

def delete_document_service(document_id: str) -> bool:
    """
    Delete a document, its original PDF, and its stored chunks.
    """

    # Check whether the document exists
    document = get_document(document_id)

    if document is None:
        return False

    # Build storage paths using the document ID
    pdf_path = UPLOAD_DIR / f"{document_id}.pdf"
    document_index_dir = INDEX_DIR / document_id

    # Delete the original PDF
    if pdf_path.exists():
        pdf_path.unlink()

    # Delete the document's chunks and index directory
    if document_index_dir.exists():
        shutil.rmtree(document_index_dir)

    # Delete document metadata
    return delete_document_record(document_id)