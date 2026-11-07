
import hashlib
import json
import shutil
import threading
from pathlib import Path
from uuid import UUID

import pymupdf

from backend.pdf_processing.extractor import extract_text_from_pdf
from backend.pdf_processing.chunker import chunk_text
from backend.pdf_processing.metadata import create_document_metadata

from backend.retrieval.faiss_search import create_faiss_index

from backend.storage.document_store import (
    save_document,
    get_document,
    get_all_documents,
    set_document_sha256,
    delete_document as delete_document_record
)


# Project directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Storage directories
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
INDEX_DIR = BASE_DIR / "data" / "indexes"

# Maximum PDF size: 20 MB
MAX_FILE_SIZE = 20 * 1024 * 1024

# Block size used when hashing PDF contents
HASH_BLOCK_SIZE = 1024 * 1024

DUPLICATE_MESSAGE = (
    "This PDF has already been uploaded. "
    "No duplicate document was created."
)

# Serialises the duplicate check and remembers hashes that are currently
# being processed, so two identical uploads arriving together cannot both
# create a document.
_upload_guard = threading.Lock()
_pending_hashes = set()


class DuplicateDocumentError(ValueError):
    """Raised when an uploaded PDF already exists in the library."""

    def __init__(self, existing_document: dict | None = None):
        super().__init__(DUPLICATE_MESSAGE)
        self.existing_document = existing_document


def calculate_sha256(pdf_bytes: bytes) -> str:
    """Return the SHA-256 hex digest of PDF bytes, hashed block by block."""

    digest = hashlib.sha256()
    view = memoryview(pdf_bytes)

    for start in range(0, len(view), HASH_BLOCK_SIZE):
        digest.update(view[start:start + HASH_BLOCK_SIZE])

    return digest.hexdigest()


def _calculate_file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file without loading it whole."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(HASH_BLOCK_SIZE), b""):
            digest.update(block)

    return digest.hexdigest()


def _find_document_by_hash(sha256: str) -> dict | None:
    """
    Find a stored document whose PDF has the given SHA-256 hash.

    Documents saved before hashing existed have no "sha256" field. Their
    hash is calculated once from the original stored PDF and saved on the
    record. If that PDF is missing or unreadable, the document is skipped;
    no hash is ever invented for it.
    """

    for document in get_all_documents():
        document_id = document.get("id")
        stored_hash = document.get("sha256")

        if not stored_hash:
            try:
                UUID(document_id)
                pdf_path = UPLOAD_DIR / f"{document_id}.pdf"
                stored_hash = _calculate_file_sha256(pdf_path)
            except (ValueError, TypeError, AttributeError, OSError):
                continue

            set_document_sha256(document_id, stored_hash)

        if isinstance(stored_hash, str) and stored_hash.lower() == sha256:
            return document

    return None


def upload_document(
    filename: str,
    pdf_bytes: bytes,
    chunk_size: int = 1000
) -> dict:
    """
    Upload and process a PDF document.

    Processing steps:
    1. Validate the uploaded PDF and reject duplicates (SHA-256 of the
       file contents) before any extraction or embedding work.
    2. Extract text from the PDF.
    3. Divide text into chunks.
    4. Generate document metadata.
    5. Save the original PDF.
    6. Save extracted chunks.
    7. Generate embeddings and create a FAISS index.
    8. Save document metadata.

    Returns the completed document metadata.
    Raises DuplicateDocumentError if the same PDF is already in the library.
    """

    # Validate filename
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("Filename is required.")

    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")

    # Validate PDF content
    if not isinstance(pdf_bytes, bytes) or not pdf_bytes:
        raise ValueError("The uploaded PDF is empty or invalid.")

    if len(pdf_bytes) > MAX_FILE_SIZE:
        raise ValueError("PDF size must not exceed 20 MB.")

    # Validate chunk size
    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
    ):
        raise ValueError("Chunk size must be a positive integer.")

    # Detect duplicates by file contents, not by filename
    sha256 = calculate_sha256(pdf_bytes)

    with _upload_guard:
        existing_document = _find_document_by_hash(sha256)

        if existing_document is not None:
            raise DuplicateDocumentError(existing_document)

        # An identical PDF is already being processed by another request
        if sha256 in _pending_hashes:
            raise DuplicateDocumentError()

        _pending_hashes.add(sha256)

    try:
        return _process_new_document(
            filename=filename,
            pdf_bytes=pdf_bytes,
            chunk_size=chunk_size,
            sha256=sha256
        )

    finally:
        with _upload_guard:
            _pending_hashes.discard(sha256)


def _process_new_document(
    filename: str,
    pdf_bytes: bytes,
    chunk_size: int,
    sha256: str
) -> dict:
    """Extract, chunk, index and store a PDF that is not a duplicate."""

    # Extract text from PDF
    extracted_pages = extract_text_from_pdf(pdf_bytes)

    # Generate text chunks
    chunks = chunk_text(
        extracted_pages,
        chunk_size=chunk_size
    )

    if not chunks:
        raise ValueError("No readable text chunks were generated.")

    # Get total PDF pages
    pdf_document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    try:
        total_pages = len(pdf_document)
    finally:
        pdf_document.close()

    # Create document metadata
    metadata = create_document_metadata(
        filename=filename.strip(),
        file_size=len(pdf_bytes),
        total_pages=total_pages,
        total_chunks=len(chunks),
        sha256=sha256
    )

    document_id = metadata["id"]

    # Create storage paths
    pdf_path = UPLOAD_DIR / f"{document_id}.pdf"
    document_index_dir = INDEX_DIR / document_id
    chunks_path = document_index_dir / "chunks.json"
    faiss_index_path = document_index_dir / "faiss.index"

    try:
        # Create storage directories
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        document_index_dir.mkdir(parents=True, exist_ok=True)

        # Save original PDF
        pdf_path.write_bytes(pdf_bytes)

        # Save extracted chunks
        with chunks_path.open("w", encoding="utf-8") as file:
            json.dump(
                chunks,
                file,
                indent=4,
                ensure_ascii=False
            )

        # Generate embeddings and create FAISS index
        indexed_chunks = create_faiss_index(
            document_id=document_id,
            chunks=chunks
        )

        # Verify indexing
        if indexed_chunks != len(chunks):
            raise RuntimeError(
                "FAISS indexing failed: chunk count mismatch."
            )

        # Verify index file exists
        if not faiss_index_path.is_file():
            raise RuntimeError(
                "FAISS index file was not created."
            )

        # Add storage paths to metadata
        metadata["file_path"] = (
            pdf_path.relative_to(BASE_DIR).as_posix()
        )

        metadata["chunks_path"] = (
            chunks_path.relative_to(BASE_DIR).as_posix()
        )

        metadata["faiss_index_path"] = (
            faiss_index_path.relative_to(BASE_DIR).as_posix()
        )

        # Save metadata only after successful processing
        save_document(metadata)

    except Exception:
        # Remove partially saved files if processing fails
        if pdf_path.exists():
            pdf_path.unlink()

        if document_index_dir.exists():
            shutil.rmtree(document_index_dir)

        raise

    return metadata


def delete_document_service(document_id: str) -> bool:
    """
    Delete a document, its original PDF,
    its chunks, its FAISS index, and its metadata.
    """

    # Validate document ID
    if not isinstance(document_id, str):
        raise ValueError("Invalid document ID.")

    try:
        # Our generated document IDs are UUIDs
        UUID(document_id)
    except ValueError:
        raise ValueError("Invalid document ID.")

    # Check whether the document exists
    document = get_document(document_id)

    if document is None:
        return False

    # Build storage paths using the validated ID
    pdf_path = UPLOAD_DIR / f"{document_id}.pdf"
    document_index_dir = INDEX_DIR / document_id

    # Delete original PDF
    if pdf_path.exists():
        pdf_path.unlink()

    # Delete chunks and FAISS index
    if document_index_dir.exists():
        shutil.rmtree(document_index_dir)

    # Delete document metadata
    return delete_document_record(document_id)
