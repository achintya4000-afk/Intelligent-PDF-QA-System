
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

# Project configuration
APP_NAME = "DocuMind"
APP_VERSION = "1.0.0"

# Validate API key
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing. Please check your root .env file."
    )
