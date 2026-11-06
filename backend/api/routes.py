from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field

from backend.services.chat_service import process_chat_question

from backend.services.document_service import (
    upload_document,
    delete_document_service
)

from backend.services.reindex_service import (
    reindex_existing_documents
)

from backend.storage.document_store import get_all_documents


router = APIRouter()


# Chat request model
class ChatRequest(BaseModel):
    document_id: str
    conversation_id: str | None = None
    history: list[dict] = Field(default_factory=list)
    question: str
    top_k: int = Field(default=4, ge=1, le=20)
    show_sources: bool = True


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


# Re-index existing documents
@router.post("/api/documents/reindex")
async def reindex_documents():
    try:
        result = reindex_existing_documents()

        return {
            "message": "Document re-indexing process completed.",
            "summary": result
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to complete document re-indexing."
        )


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


# Chat with an uploaded PDF
@router.post("/api/chat")
async def chat_with_document(request: ChatRequest):
    try:
        result = process_chat_question(
            document_id=request.document_id,
            question=request.question,
            top_k=request.top_k,
            history=request.history
        )

        sources = []

        if request.show_sources:
            for source in result["sources"]:
                sources.append({
                    "source_id": source["source_id"],
                    "chunk_id": source.get("chunk_id"),
                    "page": source["page"],
                    "text": source["text"],
                    "title": f"Page {source['page']}"
                })

        return {
            "answer": result["answer"],
            "sources": sources,
            "supported": result["supported"]
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except RuntimeError:
        raise HTTPException(
            status_code=502,
            detail="Unable to generate an answer. Please try again."
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your question."
        )