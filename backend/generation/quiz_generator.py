"""Generate document-grounded multiple-choice quizzes from uploaded PDFs."""

import json
import logging
import re
import time
from pathlib import Path
from uuid import UUID

from google.genai import types

from backend.config import GEMINI_MODEL
from backend.generation.answer_generator import get_gemini_client
from backend.storage.document_store import get_document

logger = logging.getLogger("uvicorn.error")

BASE_DIR = Path(__file__).resolve().parents[2]
INDEX_DIR = BASE_DIR / "data" / "indexes"
VALID_COUNTS = {5, 10, 15, 20}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
MAX_SOURCE_CHARACTERS = 24000


def _load_quiz_source(document_id: str) -> tuple[dict, list[dict]]:
    """Load and validate page-labelled chunks for a stored document."""
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("Please select a PDF document.")

    try:
        UUID(document_id)
    except (ValueError, AttributeError):
        raise ValueError("Invalid document ID.") from None

    document = get_document(document_id)
    if document is None:
        raise FileNotFoundError("The selected document was not found.")

    chunks_path = (INDEX_DIR / document_id / "chunks.json").resolve()
    try:
        chunks_path.relative_to(INDEX_DIR.resolve())
    except ValueError:
        raise ValueError("Invalid document storage location.") from None

    if not chunks_path.is_file():
        raise FileNotFoundError(
            "The document text is missing. Please re-index the PDF and try again."
        )

    try:
        with chunks_path.open("r", encoding="utf-8") as file:
            stored_chunks = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("Unable to read the selected PDF's extracted text.") from error

    if not isinstance(stored_chunks, list):
        raise RuntimeError("The selected PDF has an invalid text index.")

    chunks = []
    for item in stored_chunks:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        page = item.get("page")
        if (
            isinstance(text, str) and text.strip()
            and isinstance(page, int) and not isinstance(page, bool)
            and page >= 1
        ):
            chunks.append({"page": page, "text": text.strip()})

    if not chunks:
        raise ValueError(
            "No readable text is available in this PDF. Please upload a text-based PDF."
        )

    return document, chunks


def _select_source_chunks(chunks: list[dict]) -> list[dict]:
    """Keep coverage across the document within a bounded prompt size."""
    total_characters = sum(len(chunk["text"]) + 24 for chunk in chunks)
    if total_characters <= MAX_SOURCE_CHARACTERS:
        return chunks

    # Choose evenly spaced chunks so later chapters are represented too.
    average_size = max(1, total_characters // len(chunks))
    target_count = max(1, MAX_SOURCE_CHARACTERS // average_size)
    target_count = min(len(chunks), target_count)

    if target_count == 1:
        indexes = [0]
    else:
        indexes = sorted({
            round(position * (len(chunks) - 1) / (target_count - 1))
            for position in range(target_count)
        })

    selected = []
    used = 0
    for index in indexes:
        chunk = chunks[index]
        remaining = MAX_SOURCE_CHARACTERS - used - 24
        if remaining <= 0:
            break
        text = chunk["text"][:remaining]
        if text:
            selected.append({"page": chunk["page"], "text": text})
            used += len(text) + 24

    return selected


def _build_prompt(filename: str, chunks: list[dict], count: int, difficulty: str) -> str:
    source = "\n\n".join(
        f"[PDF PAGE {chunk['page']}]\n{chunk['text']}" for chunk in chunks
    )
    difficulty_rules = {
        "easy": "Test direct recall of definitions, facts, and clearly stated concepts.",
        "medium": "Test understanding, distinctions, and straightforward application of concepts.",
        "hard": "Test deeper understanding, relationships, and careful application of the document's concepts. Do not require facts absent from the PDF.",
    }
    return f"""You are DocuMind's document-grounded quiz generator.
Create exactly {count} multiple-choice questions using ONLY the PDF excerpts below.

DOCUMENT: {filename}
DIFFICULTY: {difficulty.upper()}
{difficulty_rules[difficulty]}

STRICT SOURCE RULES:
- Every question, all four options, the correct answer, and explanation must be supported by the supplied PDF excerpts.
- Do not use outside knowledge, even if it is factually correct.
- Do not invent details or assume information not stated in the excerpts.
- Cover different topics across the available pages where possible. Avoid duplicate or near-duplicate questions.
- Each question must have exactly four concise, distinct options and exactly one correct option.
- Wrong options must be plausible but clearly contradicted or unsupported by the source.
- The explanation must briefly explain why the correct answer is supported by the PDF.
- source_pages must list the actual PDF page number(s) supporting that question. Use only page numbers shown in the excerpts.
- Do not include answers in the question text.

Return ONLY valid JSON, with this exact structure:
{{
  "questions": [
    {{
      "question": "Question text",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "answer_index": 0,
      "explanation": "Source-grounded explanation.",
      "source_pages": [1]
    }}
  ]
}}

The answer_index is zero-based (0 to 3). Do not use Markdown fences or add text outside the JSON.

PDF EXCERPTS:
{source}
""".strip()


def _parse_json_response(response_text: str) -> dict:
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError("The AI returned an empty quiz response.")
    text = response_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("The AI returned invalid quiz data.") from None
        try:
            payload = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            raise ValueError("The AI returned invalid quiz data.") from None
    if not isinstance(payload, dict):
        raise ValueError("The AI returned an invalid quiz structure.")
    return payload


def _validate_questions(payload: dict, requested_count: int, valid_pages: set[int]) -> list[dict]:
    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list) or len(raw_questions) != requested_count:
        raise ValueError("The AI did not generate the requested number of questions. Please try again.")

    validated = []
    seen_questions = set()
    for index, item in enumerate(raw_questions):
        if not isinstance(item, dict):
            raise ValueError("The AI returned an invalid question. Please try again.")

        question = item.get("question")
        options = item.get("options")
        answer_index = item.get("answer_index")
        explanation = item.get("explanation")
        pages = item.get("source_pages")

        if not isinstance(question, str) or not question.strip():
            raise ValueError("A generated question was empty. Please try again.")
        if not isinstance(options, list) or len(options) != 4:
            raise ValueError("A generated question did not contain four options. Please try again.")
        clean_options = [option.strip() for option in options if isinstance(option, str) and option.strip()]
        if len(clean_options) != 4 or len({value.casefold() for value in clean_options}) != 4:
            raise ValueError("A generated question contained invalid or duplicate options. Please try again.")
        if not isinstance(answer_index, int) or isinstance(answer_index, bool) or not 0 <= answer_index < 4:
            raise ValueError("A generated question had an invalid answer key. Please try again.")
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError("A generated question was missing its explanation. Please try again.")
        if not isinstance(pages, list):
            raise ValueError("A generated question was missing its PDF page reference. Please try again.")

        clean_pages = sorted({page for page in pages if isinstance(page, int) and not isinstance(page, bool) and page in valid_pages})
        if not clean_pages:
            raise ValueError("A generated question had no valid supporting PDF page. Please try again.")

        normalized_question = re.sub(r"\s+", " ", question).strip().casefold()
        if normalized_question in seen_questions:
            raise ValueError("The AI generated duplicate questions. Please try again.")
        seen_questions.add(normalized_question)

        validated.append({
            "question": question.strip(),
            "options": clean_options,
            "answer_index": answer_index,
            "explanation": explanation.strip(),
            "source_pages": clean_pages,
        })

    return validated


def generate_document_quiz(document_id: str, count: int = 5, difficulty: str = "medium") -> dict:
    """Generate a validated, source-grounded multiple-choice quiz."""
    if not isinstance(count, int) or isinstance(count, bool) or count not in VALID_COUNTS:
        raise ValueError("Question count must be 5, 10, 15, or 20.")
    if difficulty not in VALID_DIFFICULTIES:
        raise ValueError("Difficulty must be easy, medium, or hard.")

    document, all_chunks = _load_quiz_source(document_id)
    chunks = _select_source_chunks(all_chunks)
    valid_pages = {chunk["page"] for chunk in chunks}
    prompt = _build_prompt(document.get("filename", "Uploaded PDF"), chunks, count, difficulty)

    last_error = None
    client = get_gemini_client()
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=min(8192, 500 + count * 260),
                    response_mime_type="application/json",
                ),
            )
            payload = _parse_json_response(response.text)
            questions = _validate_questions(payload, count, valid_pages)
            return {
                "document_id": document_id,
                "filename": document.get("filename", "Uploaded PDF"),
                "difficulty": difficulty,
                "count": count,
                "questions": questions,
            }
        except ValueError as error:
            last_error = error
            logger.warning("Quiz generation attempt %d returned invalid data: %s", attempt + 1, error)
        except Exception as error:
            last_error = error
            logger.warning("Quiz generation attempt %d failed: %r", attempt + 1, error)

        if attempt == 0:
            time.sleep(1)

    raise RuntimeError("Unable to generate a valid quiz from this PDF. Please try again.") from last_error
