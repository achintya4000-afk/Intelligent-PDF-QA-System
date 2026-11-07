
from datetime import datetime, timezone
from uuid import uuid4


def create_document_metadata(
    filename: str,
    file_size: int,
    total_pages: int,
    total_chunks: int,
    sha256: str | None = None
) -> dict:
    """
    Create metadata for an uploaded PDF document.

    "sha256" is the SHA-256 hex digest of the PDF's file contents. It is
    used to detect duplicate uploads and is stored only when provided.
    """

    metadata = {
        "id": str(uuid4()),
        "filename": filename,
        "size": file_size,
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready"
    }

    if sha256:
        metadata["sha256"] = sha256

    return metadata
