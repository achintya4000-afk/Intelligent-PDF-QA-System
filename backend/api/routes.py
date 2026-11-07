import logging
import re
from typing import Literal

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.services.chat_service import process_chat_question
from backend.generation.summarizer import generate_document_summary
from backend.generation.quiz_generator import generate_document_quiz

from backend.services.document_service import (
    upload_document,
    delete_document_service,
    DuplicateDocumentError
)

from backend.services.reindex_service import (
    reindex_existing_documents
)

from backend.storage.document_store import get_all_documents


router = APIRouter()

# Uses uvicorn's own logger so errors appear in the terminal where
# the backend is running. The browser only receives a short message.
logger = logging.getLogger("uvicorn.error")


# Chat request model
#
# "mode" selects how the answer is produced:
#   auto - search the selected PDF first, fall back to general AI
#          (with Google Search Grounding for current-information questions)
#   pdf  - answer only from the selected PDF (never uses the web)
#   ai   - general AI only, the PDF is never searched (Google Search
#          Grounding for current-information questions)
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
        if request.show_sources and result["answer_type"] in (
            "pdf", "hybrid", "pdf_web"
        ):
            for source in result["sources"]:
                sources.append({
                    "source_id": source["source_id"],
                    "chunk_id": source.get("chunk_id"),
                    "page": source["page"],
                    "text": source["text"],
                    "title": f"Page {source['page']}"
                })

        answer_text = result["answer"]

        # Web sources (Google Search Grounding). Only sources that Gemini
        # returned in its grounding metadata are ever present here.
        web_sources = []

        if request.show_sources and result["answer_type"] in ("web", "pdf_web"):
            for source in result.get("web_sources", []):
                web_sources.append({
                    "source_id": source["source_id"],
                    "title": source["title"],
                    "url": source["url"],
                    "domain": source["domain"],
                    "snippets": source["snippets"],
                    "cited": source["cited"]
                })

        # With sources hidden, the [W1] markers would point to nothing.
        if not request.show_sources:
            answer_text = re.sub(r"\s?\[W\d+\]", "", answer_text)

        return {
            "answer": answer_text,
            "sources": sources,
            # True only when the answer is grounded in PDF evidence.
            "supported": result["supported"],
            # The mode that was requested: "auto", "pdf" or "ai".
            "mode": result["mode"],
            # Where the answer really came from: "pdf", "general_ai", or
            # "hybrid" (Auto: cited PDF part plus a labelled general AI part).
            "answer_type": result["answer_type"],
            # Auto mode only: why the PDF was not used, otherwise null.
            "fallback_reason": result["fallback_reason"],
            # Web answers: "web" or "pdf_web" are new answer_type values.
            # Real web sources only; empty when the web was not used.
            "web_sources": web_sources,
            # {"status": "used" | "no_sources" | "failed" | "not_used",
            #  "queries": [...]}
            "web_search": result.get(
                "web_search", {"status": "not_used", "queries": []}
            )
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