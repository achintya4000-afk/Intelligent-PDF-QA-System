
import os
from pathlib import Path
from dotenv import load_dotenv

# Find the root project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from root .env
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE)

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")



def _float_env(name: str, default: float) -> float:
    """
    Read an optional float from the environment.

    Invalid or out-of-range values (outside 0..1) fall back to the default,
    so a typo in .env can never break start-up.
    """

    raw_value = os.getenv(name)

    if raw_value is None or not raw_value.strip():
        return default

    try:
        value = float(raw_value)
    except ValueError:
        return default

    return value if 0.0 <= value <= 1.0 else default


# Auto response mode: relevance gate for PDF evidence.
#
# FAISS scores are cosine similarities of normalised all-MiniLM-L6-v2
# vectors (range -1..1). RRF scores only encode rank and BM25 scores depend
# on the document, so neither can say "relevant" on its own.
# These values are a coarse first filter only; the final decision is made by
# grounded generation that must return valid citations.
# Override in .env if your documents need different values.

# Below this best cosine score nothing in the PDF is semantically close.
AUTO_MIN_SEMANTIC_SCORE = _float_env("AUTO_MIN_SEMANTIC_SCORE", 0.20)

# At or above this score the semantic match alone is accepted.
# Between the two values an exact keyword match is also required.
AUTO_CONFIDENT_SEMANTIC_SCORE = max(
    _float_env("AUTO_CONFIDENT_SEMANTIC_SCORE", 0.35),
    AUTO_MIN_SEMANTIC_SCORE
)

# Share of the question's keywords one retrieved chunk must contain
# when the semantic score is only in the in-between range.
AUTO_MIN_LEXICAL_COVERAGE = _float_env("AUTO_MIN_LEXICAL_COVERAGE", 0.60)

# Project configuration
APP_NAME = "DocuMind"
APP_VERSION = "1.0.0"

# Validate API key
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing. Please check your root .env file."
    )
