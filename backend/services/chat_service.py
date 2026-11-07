from uuid import UUID

from backend.storage.document_store import get_document
from backend.retrieval.retriever import retrieve_relevant_chunks
from backend.generation.answer_generator import generate_answer


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


def process_chat_question(
    document_id: str,
    question: str,
    top_k: int = 4,
    history: list = None
) -> dict:
    """
    Process a user's PDF-related question.

    Flow:
        1. Validate document ID and question.
        2. Validate previous conversation history.
        3. Check whether the document exists.
        4. Build a contextual retrieval query when needed.
        5. Retrieve relevant PDF chunks using hybrid retrieval.
        6. Generate an answer using Gemini and conversation context.
        7. Return the answer and cited source references.
    """

    # Validate document ID.
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("A valid document ID is required.")

    try:
        UUID(document_id)
    except ValueError as error:
        raise ValueError("Invalid document ID format.") from error

    document_id = document_id.strip()

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

    # Check whether the document exists.
    document = get_document(document_id)

    if document is None:
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

    # For a follow-up, tell the answer generator which earlier
    # question the pronouns ("it", "this") refer to.
    follow_up_of = None

    if conversation_history and is_follow_up_question(question):
        follow_up_of = find_anchor_question(conversation_history)

    # Generate an answer using Gemini and conversation history.
    result = generate_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
        history=conversation_history,
        follow_up_of=follow_up_of
    )

    return {
        "document_id": document_id,
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"],
        "supported": result["supported"]
    }