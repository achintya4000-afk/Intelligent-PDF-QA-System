
from sentence_transformers import SentenceTransformer
import numpy as np


# Pretrained embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load the model once and reuse it
_model = None


def get_embedding_model():
    """
    Load the Sentence Transformer model only once.
    """

    global _model

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    return _model


def generate_embeddings(texts: list) -> np.ndarray:
    """
    Convert a list of text passages into numerical embeddings.

    Returns a NumPy array of shape:
    (number_of_texts, 384)
    """

    if not isinstance(texts, list):
        raise ValueError("Input must be a list of texts.")

    if not texts:
        raise ValueError("At least one text is required.")

    if not all(isinstance(text, str) and text.strip() for text in texts):
        raise ValueError("Every item must be a non-empty string.")

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return np.asarray(embeddings, dtype=np.float32)


def generate_single_embedding(text: str) -> np.ndarray:
    """
    Generate an embedding for a single text.
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must be a non-empty string.")

    embeddings = generate_embeddings([text])

    return embeddings[0]
