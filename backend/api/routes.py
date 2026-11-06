
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from backend.services.document_service import (
    upload_document,
    delete_document_service
)
from backend.storage.document_store import get_all_documents


router = APIRouter()


# Health check
@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "DocuMind backend is running successfully"
    }


# Get all uploaded documents
@router.get("/api/documents")
async def list_documents():

    documents = get_all_documents()

    public_documents = []

    for document in documents:
        public_documents.append({
            "id": document["id"],
            "filename": document["filename"],
            "size": document["size"],
            "uploaded_at": document["uploaded_at"],
            "status": document["status"]
        })

    return {
        "documents": public_documents
    }


# Upload and process a PDF
@router.post("/api/documents/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    chunk_size: int = Form(1000)
):

    try:
        pdf_bytes = await file.read()

        document = upload_document(
            filename=file.filename or "",
            pdf_bytes=pdf_bytes,
            chunk_size=chunk_size
        )

        public_document = {
            "id": document["id"],
            "filename": document["filename"],
            "size": document["size"],
            "uploaded_at": document["uploaded_at"],
            "status": document["status"]
        }

        return {
            "document": public_document
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    finally:
        await file.close()


# Delete a document
@router.delete("/api/documents/{document_id}")
async def remove_document(document_id: str):

    try:
        deleted = delete_document_service(document_id)

        if not deleted:
            raise HTTPException(
                status_code=404,
                detail="Document not found."
            )

        return {
            "message": "Document deleted successfully.",
            "document_id": document_id
        }

    except OSError:
        raise HTTPException(
            status_code=500,
            detail="Unable to delete document files."
        )
