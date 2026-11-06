
import json
from pathlib import Path
from uuid import UUID

import faiss

from backend.storage.document_store import get_all_documents
from backend.retrieval.faiss_search import create_faiss_index


# Project directories
BASE_DIR = Path(__file__).resolve().parents[2]
INDEX_DIR = BASE_DIR / "data" / "indexes"


def reindex_existing_documents() -> dict:
    """
    Check all stored documents and create FAISS indexes
    for documents whose indexes are missing or invalid.

    Existing valid indexes are skipped.
    """

    documents = get_all_documents()

    summary = {
        "total_documents": len(documents),
        "indexed": 0,
        "skipped": 0,
        "failed": 0,
        "results": []
    }

    for document in documents:

        document_id = document.get("id")
        filename = document.get("filename", "Unknown")

        # Validate document ID
        try:
            if not isinstance(document_id, str):
                raise ValueError("Invalid document ID.")

            UUID(document_id)

        except (ValueError, TypeError, AttributeError):
            summary["failed"] += 1

            summary["results"].append({
                "document_id": document_id,
                "filename": filename,
                "status": "failed",
                "message": "Invalid document ID."
            })

            continue

        # Locate the stored chunks and FAISS index
        document_index_dir = INDEX_DIR / document_id
        chunks_path = document_index_dir / "chunks.json"
        faiss_index_path = document_index_dir / "faiss.index"

        try:

            # Existing chunks are required to rebuild an index
            if not chunks_path.is_file():
                raise FileNotFoundError(
                    "chunks.json not found. This document cannot be "
                    "re-indexed from stored chunks."
                )

            # Load stored chunks
            with chunks_path.open("r", encoding="utf-8") as file:
                chunks = json.load(file)

            if not isinstance(chunks, list) or not chunks:
                raise ValueError("No valid chunks are available.")

            # Check that every chunk has the required fields
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    raise ValueError("Invalid chunk format.")

                if not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
                    raise ValueError("A chunk contains empty or invalid text.")

                if "chunk_id" not in chunk or "page" not in chunk:
                    raise ValueError("A chunk is missing required metadata.")

            # Check whether a valid FAISS index already exists
            if faiss_index_path.is_file():

                try:
                    existing_index = faiss.read_index(
                        str(faiss_index_path)
                    )

                    if existing_index.ntotal == len(chunks):
                        summary["skipped"] += 1

                        summary["results"].append({
                            "document_id": document_id,
                            "filename": filename,
                            "status": "skipped",
                            "message": "A valid FAISS index already exists."
                        })

                        continue

                except Exception:
                    # A corrupted index will be rebuilt below.
                    pass

            # Create a new FAISS index
            indexed_chunks = create_faiss_index(
                document_id=document_id,
                chunks=chunks
            )

            if indexed_chunks != len(chunks):
                raise RuntimeError(
                    "The indexed vector count does not match "
                    "the number of stored chunks."
                )

            # Verify the newly created index
            if not faiss_index_path.is_file():
                raise RuntimeError("FAISS index file was not created.")

            new_index = faiss.read_index(str(faiss_index_path))

            if new_index.ntotal != len(chunks):
                raise RuntimeError(
                    "FAISS index verification failed."
                )

            summary["indexed"] += 1

            summary["results"].append({
                "document_id": document_id,
                "filename": filename,
                "status": "indexed",
                "message": "FAISS index created and verified.",
                "total_chunks": len(chunks)
            })

        except Exception as error:

            summary["failed"] += 1

            summary["results"].append({
                "document_id": document_id,
                "filename": filename,
                "status": "failed",
                "message": str(error)
            })

    return summary
