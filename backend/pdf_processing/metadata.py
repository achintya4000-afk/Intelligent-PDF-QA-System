
from datetime import datetime, timezone
from uuid import uuid4


def create_document_metadata(
    filename: str,
    file_size: int,
    total_pages: int,
    total_chunks: int
) -> dict:
    """
    Create metadata for an uploaded PDF document.
    """

    return {
        "id": str(uuid4()),
        "filename": filename,
        "size": file_size,
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready"
    }
