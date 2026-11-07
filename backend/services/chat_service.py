from uuid import UUID

from backend.storage.document_store import get_document
from backend.retrieval.retriever import retrieve_relevant_chunks
from backend.generation.answer_generator import generate_answer
from backend.generation.general_answer_generator import (
    generate_general_answer
)
from backend.generation.hybrid_answer_generator import (
    generate_hybrid_answer
)
from backend.generation.evidence_checker import assess_retrieval_evidence


# Supported response modes (the exact values the API accepts).
MODE_AUTO = "auto"    # PDF first, general AI when the PDF has no evidence
MODE_PDF = "pdf"      # PDF evidence only
MODE_AI = "ai"        # General AI only, the PDF is never searched

RESPONSE_MODES = (MODE_AUTO, MODE_PDF, MODE_AI)

# Values of "answer_type" in the result.
ANSWER_TYPE_PDF = "pdf"
ANSWER_TYPE_GENERAL_AI = "general_ai"
# PDF-cited answer plus a clearly labelled general AI section (Auto only).
ANSWER_TYPE_HYBRID = "hybrid"

# Why Auto mode answered without the PDF ("fallback_reason").
FALLBACK_NO_PDF = "no_pdf_selected"
FALLBACK_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
FALLBACK_UNVERIFIED_EVIDENCE = "unverified_evidence"


# Words that show a short question is about the document itself,
# not about the previous answer ("What does this document say...").
DOCUMENT_REFERENCE_WORDS = (
    "document",
    "pdf",
    "file",
    "text",
    "page",
    "chapter",
    "section",
    "passage",
    "paper"
)


def validate_history(history: list) -> list:
    """
    Validate and limit conversation history.

    Only valid user and assistant messages are retained.
    The latest 8 messages are used to provide context.
    """

    if not isinstance(history, list):
        raise ValueError("Conversation history must be a list.")

    valid_history = []

    for message in history[-8:]:

        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = message.get("content")

        if role not in ("user", "assistant"):
            continue

        if not isinstance(content, str) or not content.strip():
            continue

        valid_history.append({
            "role": role,
            "content": content.strip()[:3000]
        })

    return valid_history


def is_follow_up_question(question: str) -> bool:
    """
    Identify common follow-up questions that need previous context.

    This is a simple heuristic, not a perfect language classifier.
    """

    question_lower = question.lower().strip()

    follow_up_phrases = (
        "explain it",
        "explain this",
        "explain that",
        "describe it",
        "describe this",
        "tell me more",
        "more details",
        "in simple words",
        "in simpler words",
        "give an example",
        "give me an example",
        "what about it",
        "what about its",
        "how does it",
        "why is it",
        "why does it",
        "what is its",
        "what are its",
        "compare them",
        "difference between them",
        "summarize it",
        "summarise it",
        "elaborate",
        "can you elaborate",
        "what else",
        "go on",
        "continue",
        "and what about",
        "how so"
    )

    if any(
        question_lower.startswith(phrase)
        for phrase in follow_up_phrases
    ):
        return True

    short_follow_up_words = (
        "it",
        "this",
        "that",
        "they",
        "them",
        "their",
        "its"
    )

    words = [
        word.strip(".,?!;:\"'()")
        for word in question_lower.split()
    ]

    if len(words) <= 7:

        # "What does this document say about DNS?" is a new question
        # about the document, not a follow-up to the previous answer.
        if any(word in DOCUMENT_REFERENCE_WORDS for word in words):
            return False

        return any(
            word in short_follow_up_words
            for word in words
        )

    return False


def find_anchor_question(history: list):
    """
    Find the question that the current follow-up refers to.

    Chained follow-ups ("What is TCP?" -> "Explain it in simple words"
    -> "Give me an example") must all point back to the original
    question. So the most recent user question that is NOT itself a
    follow-up is used. If every earlier question is a follow-up, the
    latest one is used as a fallback.
    """

    previous_user_questions = [
        message["content"]
        for message in history
        if message["role"] == "user"
    ]

    if not previous_user_questions:
        return None

    for previous_question in reversed(previous_user_questions):
        if not is_follow_up_question(previous_question):
            return previous_question

    return previous_user_questions[-1]


def build_retrieval_query(
    question: str,
    history: list
) -> str:
    """
    Build a retrieval query.

    Independent questions use only the current question.
    Follow-up questions are combined with the question they refer to.
    """

    if not history:
        return question

    if not is_follow_up_question(question):
        return question

    anchor_question = find_anchor_question(history)

    if not anchor_question:
        return question

    return f"{anchor_question[:500]} {question}"


def validate_mode(mode) -> str:
    """
    Validate the response mode.

    Only the exact values "auto", "pdf" and "ai" are accepted.
    """

    if not isinstance(mode, str) or mode not in RESPONSE_MODES:
        raise ValueError(
            "Unsupported response mode. "
            "Use one of: auto, pdf, ai."
        )

    return mode


def validate_document_id(document_id) -> str:
    """
    Validate the format of a document ID and return it cleaned.
    """

    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("A valid document ID is required.")

    document_id = document_id.strip()

    try:
        UUID(document_id)
    except ValueError as error:
        raise ValueError("Invalid document ID format.") from error

    return document_id


def build_result(
    mode: str,
    answer_type: str,
    result: dict,
    document_id: str | None,
    question: str,
    fallback_reason: str | None = None
) -> dict:
    """
    Build the chat result in one consistent structure.

    General AI answers never carry PDF sources, whatever the caller
    passes in.
    """

    # Hybrid answers contain real PDF citations in their PDF part.
    is_pdf_answer = answer_type in (ANSWER_TYPE_PDF, ANSWER_TYPE_HYBRID)

    return {
        "document_id": document_id if is_pdf_answer else None,
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"] if is_pdf_answer else [],
        "supported": bool(result["supported"]) if is_pdf_answer else False,
        "mode": mode,
        "answer_type": answer_type,
        "fallback_reason": fallback_reason
    }


def process_chat_question(
    document_id: str | None,
    question: str,
    top_k: int = 4,
    history: list = None,
    mode: str = MODE_AUTO
) -> dict:
    """
    Process a user's question in one of three response modes.

    Modes:
        pdf   Retrieve from the selected PDF and answer only from that
              evidence (the original DocuMind behaviour).
        ai    Answer with Gemini's general knowledge. No retrieval of any
              kind is performed and document_id is ignored.
        auto  Retrieve from the selected PDF first. When the evidence is
              sufficient and Gemini cites it, answer from the PDF and, when
              useful, add a clearly labelled general AI section ("hybrid").
              Otherwise answer with general knowledge and say so.

    Returns:
        answer, sources, supported, mode, answer_type ("pdf", "hybrid" or
        "general_ai") and fallback_reason (Auto mode only).
    """

    mode = validate_mode(mode)

    # Validate question.
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must not be empty.")

    question = question.strip()

    # Validate top_k.
    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k <= 0
    ):
        raise ValueError("top_k must be a positive integer.")

    # Validate conversation history.
    conversation_history = validate_history(
        history if history is not None else []
    )

    # For a follow-up, tell the answer generator which earlier
    # question the pronouns ("it", "this") refer to.
    follow_up_of = None

    if conversation_history and is_follow_up_question(question):
        follow_up_of = find_anchor_question(conversation_history)

    has_document = (
        isinstance(document_id, str) and bool(document_id.strip())
    )

    # ------------------------------------------------------------------
    # Only AI: the PDF is not searched and not even looked up.
    # ------------------------------------------------------------------
    if mode == MODE_AI:
        result = generate_general_answer(
            question=question,
            history=conversation_history,
            follow_up_of=follow_up_of
        )

        return build_result(
            mode, ANSWER_TYPE_GENERAL_AI, result, None, question
        )

    # ------------------------------------------------------------------
    # Auto without a PDF: nothing to search, so answer generally.
    # ------------------------------------------------------------------
    if not has_document:

        if mode == MODE_PDF:
            raise ValueError(
                "Please select or upload a PDF to use Only PDF mode."
            )

        result = generate_general_answer(
            question=question,
            history=conversation_history,
            follow_up_of=follow_up_of
        )

        return build_result(
            mode,
            ANSWER_TYPE_GENERAL_AI,
            result,
            None,
            question,
            fallback_reason=FALLBACK_NO_PDF
        )

    # ------------------------------------------------------------------
    # PDF retrieval (Only PDF and Auto).
    # ------------------------------------------------------------------
    document_id = validate_document_id(document_id)

    # Check whether the document exists.
    if get_document(document_id) is None:
        raise FileNotFoundError(
            "The requested document was not found."
        )

    # Build retrieval query without mixing unrelated questions.
    retrieval_query = build_retrieval_query(
        question=question,
        history=conversation_history
    )

    # Retrieve relevant PDF passages.
    retrieved_chunks = retrieve_relevant_chunks(
        document_id=document_id,
        query=retrieval_query,
        top_k=top_k
    )

    # Only PDF: always the PDF-grounded answer, including its own
    # "not found in the PDF" reply. No general knowledge is added.
    if mode == MODE_PDF:
        result = generate_answer(
            question=question,
            retrieved_chunks=retrieved_chunks,
            history=conversation_history,
            follow_up_of=follow_up_of
        )

        return build_result(
            mode, ANSWER_TYPE_PDF, result, document_id, question
        )

    # Auto: first decide whether the retrieved chunks are relevant.
    # Retrieval always returns "the best" chunks, relevant or not.
    evidence = assess_retrieval_evidence(
        query=retrieval_query,
        chunks=retrieved_chunks
    )

    fallback_reason = None

    if evidence["sufficient"]:

        # PDF facts with citations, plus labelled general AI detail
        # when it helps.
        result = generate_hybrid_answer(
            question=question,
            retrieved_chunks=retrieved_chunks,
            history=conversation_history,
            follow_up_of=follow_up_of
        )

        if result["supported"] and result["sources"]:
            # Gemini answered from the PDF and cited real sources.
            answer_type = (
                ANSWER_TYPE_HYBRID
                if result.get("general_added")
                else ANSWER_TYPE_PDF
            )

            return build_result(
                mode, answer_type, result, document_id, question
            )

        # Either Gemini found nothing relevant (abstained) or it wrote
        # an answer without a single valid citation. Neither may be
        # presented as a PDF answer.
        fallback_reason = (
            FALLBACK_INSUFFICIENT_EVIDENCE
            if not result["supported"]
            else FALLBACK_UNVERIFIED_EVIDENCE
        )

    else:
        fallback_reason = FALLBACK_INSUFFICIENT_EVIDENCE

    # General AI fallback: no PDF sources, no PDF citations.
    result = generate_general_answer(
        question=question,
        history=conversation_history,
        follow_up_of=follow_up_of
    )

    return build_result(
        mode,
        ANSWER_TYPE_GENERAL_AI,
        result,
        None,
        question,
        fallback_reason=fallback_reason
    )