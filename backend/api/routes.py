import logging
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from backend.services.chat_service import process_chat_question
from backend.generation.summarizer import generate_document_summary
from backend.generation.quiz_generator import generate_document_quiz
from backend.generation.glossary_generator import generate_document_glossary
from backend.generation.glossary_export import build_glossary_pdf

from backend.services.document_service import (
    UPLOAD_DIR,
    upload_document,
    delete_document_service,
    DuplicateDocumentError
)

from backend.services.reindex_service import (
    reindex_existing_documents
)

from backend.storage.document_store import get_all_documents, get_document


router = APIRouter()

# Uses uvicorn's own logger so errors appear in the terminal where
# the backend is running. The browser only receives a short message.
logger = logging.getLogger("uvicorn.error")


# Chat request model
#
# "mode" selects how the answer is produced:
#   auto - search the selected PDF first, fall back to general AI
#   pdf  - answer only from the selected PDF
#   ai   - general AI only, the PDF is never searched
# Any other value is rejected by validation (HTTP 422).
class ChatRequest(BaseModel):
    # Optional: not needed for "ai" mode, and "auto" works without a PDF.
    # "pdf" mode returns a clear 400 message when it is missing.
    document_id: str | None = None
    mode: Literal["auto", "pdf", "ai"] = "auto"
    conversation_id: str | None = None
    history: list[dict] = Field(default_factory=list)
    question: str
    top_k: int = Field(default=4, ge=1, le=20)
    show_sources: bool = True


# Smart Summarizer request model
class SummaryRequest(BaseModel):
    document_id: str
    length: str = "medium"
    format: str = "bullets"


# Quiz Generator request model
class QuizRequest(BaseModel):
    document_id: str
    count: Literal[5, 10, 15, 20] = 5
    difficulty: Literal["easy", "medium", "hard"] = "medium"


# Glossary Generator request models
class GlossaryRequest(BaseModel):
    document_id: str


class GlossaryExportRequest(BaseModel):
    document_id: str
    view: Literal["simple", "technical", "both"] = "both"
    # The glossary the user is looking at. It is validated and clipped by
    # build_glossary_pdf before anything is laid out.
    terms: list[dict]


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

        # Extraction (including OCR of scanned pages) can take a while, so it
        # runs in a worker thread and does not block the server's event loop.
        document = await run_in_threadpool(
            upload_document,
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

        response = {
            "document": public_document
        }

        # Non-fatal processing warnings (for example Gemini handwriting
        # recognition failed and PaddleOCR output was used). The document is
        # still indexed and ready; this key is absent when there are none.
        processing_warnings = document.get("processing_warnings")

        if processing_warnings:
            response["warnings"] = list(processing_warnings)

        return response

    except DuplicateDocumentError as error:
        # 409 Conflict: nothing was created or changed. "detail" keeps the
        # usual error shape so older clients still show the message.
        existing = error.existing_document

        return JSONResponse(
            status_code=409,
            content={
                "duplicate": True,
                "detail": str(error),
                "message": str(error),
                "existing_document": (
                    {
                        "id": existing.get("id"),
                        "filename": existing.get("filename")
                    }
                    if existing else None
                )
            }
        )

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


# Serve the original uploaded PDF so the browser can display it
@router.get("/api/documents/{document_id}/file")
def get_document_file(document_id: str):
    try:
        UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID.")

    if get_document(document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    pdf_path = (UPLOAD_DIR / f"{document_id}.pdf").resolve()

    try:
        pdf_path.relative_to(UPLOAD_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document storage location.")

    if not pdf_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="The original PDF file is missing. Please upload it again.",
        )

    # "inline" makes the browser display the PDF instead of downloading it.
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        headers={"Content-Disposition": 'inline; filename="document.pdf"'},
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


# Generate a summary from an uploaded PDF
@router.post("/api/summarize")
def summarize_document(request: SummaryRequest):
    """Generate a document-grounded summary using stored PDF chunks."""

    try:
        return generate_document_summary(
            document_id=request.document_id,
            length=request.length,
            output_format=request.format
        )

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

    except RuntimeError as error:
        logger.error("Summary generation failed: %s", error)
        raise HTTPException(
            status_code=502,
            detail="Unable to generate the summary. Please try again."
        )

    except Exception:
        logger.exception("Unexpected error while generating a document summary")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while generating the summary."
        )


# Generate a multiple-choice quiz grounded in an uploaded PDF
@router.post("/api/quiz/generate")
def generate_quiz(request: QuizRequest):
    try:
        return generate_document_quiz(
            document_id=request.document_id,
            count=request.count,
            difficulty=request.difficulty,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except RuntimeError as error:
        logger.error("Quiz generation failed: %s", error)
        raise HTTPException(
            status_code=502,
            detail="Unable to generate a valid quiz. Please try again.",
        )
    except Exception:
        logger.exception("Unexpected error while generating a quiz")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while generating the quiz.",
        )


# Generate a glossary of important terms from an uploaded PDF
@router.post("/api/glossary/generate")
def generate_glossary(request: GlossaryRequest):
    try:
        return generate_document_glossary(
            document_id=request.document_id
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except RuntimeError as error:
        logger.error("Glossary generation failed: %s", error)
        raise HTTPException(
            status_code=502,
            detail="Unable to generate the glossary. Please try again.",
        )
    except Exception:
        logger.exception("Unexpected error while generating a glossary")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while generating the glossary.",
        )


# Export a generated glossary as a PDF download
@router.post("/api/glossary/export")
def export_glossary(request: GlossaryExportRequest):
    try:
        pdf_bytes = build_glossary_pdf(
            document_id=request.document_id,
            terms=request.terms,
            view=request.view,
        )

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": 'attachment; filename="glossary.pdf"'
            },
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception:
        logger.exception("Unexpected error while exporting a glossary")
        raise HTTPException(
            status_code=500,
            detail="Unable to create the glossary PDF.",
        )


# Chat with an uploaded PDF or with general AI (see ChatRequest.mode)
@router.post("/api/chat")
async def chat_with_document(request: ChatRequest):
    try:
        result = process_chat_question(
            document_id=request.document_id,
            question=request.question,
            top_k=request.top_k,
            history=request.history,
            mode=request.mode
        )

        sources = []

        # General AI answers never have PDF sources.
        if request.show_sources and result["answer_type"] in ("pdf", "hybrid"):
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
            # True only when the answer is grounded in PDF evidence.
            "supported": result["supported"],
            # The mode that was requested: "auto", "pdf" or "ai".
            "mode": result["mode"],
            # Where the answer really came from: "pdf", "general_ai", or
            # "hybrid" (Auto: cited PDF part plus a labelled general AI part).
            "answer_type": result["answer_type"],
            # Auto mode only: why the PDF was not used, otherwise null.
            "fallback_reason": result["fallback_reason"]
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

    except RuntimeError as error:
        # The real reason (for example a Gemini rate limit) is written to
        # the backend terminal; the browser gets a short, safe message.
        logger.error("Answer generation failed: %s", error)

        raise HTTPException(
            status_code=502,
            detail="Unable to generate an answer. Please try again."
        )

    except Exception:
        logger.exception(
            "Unexpected error while processing a chat question"
        )

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your question."
        )