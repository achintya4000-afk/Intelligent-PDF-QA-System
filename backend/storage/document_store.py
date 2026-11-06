
import json
import threading
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parents[2]

# JSON storage location
STORAGE_DIR = BASE_DIR / "data"
METADATA_FILE = STORAGE_DIR / "documents.json"

# Prevent simultaneous writes from interfering with each other
_store_lock = threading.Lock()


def _ensure_storage():
    """Create the storage directory and JSON file if needed."""

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    if not METADATA_FILE.exists():
        METADATA_FILE.write_text("{}", encoding="utf-8")


def _read_documents():
    """Read all document records from the JSON file."""

    _ensure_storage()

    try:
        with METADATA_FILE.open("r", encoding="utf-8") as file:
            documents = json.load(file)

        if not isinstance(documents, dict):
            raise ValueError("Document storage must contain a JSON object.")

        return documents

    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Document metadata file contains invalid JSON."
        ) from error


def _write_documents(documents):
    """Save document records safely using a temporary file."""

    _ensure_storage()

    temporary_file = METADATA_FILE.with_suffix(".tmp")

    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(documents, file, indent=4, ensure_ascii=False)

    temporary_file.replace(METADATA_FILE)


def save_document(metadata: dict) -> dict:
    """Add or update a document record."""

    document_id = metadata.get("id")

    if not document_id:
        raise ValueError("Document ID is required.")

    with _store_lock:
        documents = _read_documents()
        documents[document_id] = metadata
        _write_documents(documents)

    return metadata


def get_document(document_id: str):
    """Retrieve a document using its ID."""

    with _store_lock:
        documents = _read_documents()
        return documents.get(document_id)


def get_all_documents() -> list:
    """Return all stored document records."""

    with _store_lock:
        documents = _read_documents()
        return list(documents.values())


def delete_document(document_id: str) -> bool:
    """Delete a document record using its ID."""

    with _store_lock:
        documents = _read_documents()

        if document_id not in documents:
            return False

        del documents[document_id]
        _write_documents(documents)

    return True
