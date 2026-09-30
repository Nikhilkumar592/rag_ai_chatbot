"""Central configuration: environment variables and tunable constants."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

if sys.version_info[:2] != (3, 12):
    raise RuntimeError(
        "This project requires Python 3.12. Create the venv with 'py -3.12 -m venv .venv' "
        "and run with '.\\.venv\\Scripts\\python.exe'."
    )

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "data" / "Ebook-Agentic-AI.pdf"
GDRIVE_FILE_ID = "15VLphKcY23_fpYxN62UEQRri_psRVfP9"

# --- Secrets / services ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-gemini")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# --- Models ---
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 3072  # Gemini embedding model's default output dimension
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# --- Retrieval / grounding ---
TOP_K = int(os.getenv("TOP_K", 4))
# Cosine-similarity floor: if even the best chunk is below this, the question
# is treated as out-of-scope and we refuse without calling the LLM.
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", 0.30))
# Minimum groundedness score (0-1) from the LLM grader to accept an answer.
GROUNDING_THRESHOLD = float(os.getenv("GROUNDING_THRESHOLD", 0.60))
MAX_GENERATION_ATTEMPTS = 2

REFUSAL_MESSAGE = "I cannot answer based on the provided document."


def require_keys() -> None:
    missing = [
        name
        for name, val in (
            ("GEMINI_API_KEY", GEMINI_API_KEY),
            ("PINECONE_API_KEY", PINECONE_API_KEY),
        )
        if not val
    ]
    if missing:
        if "GEMINI_API_KEY" in missing and os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "This project now uses Gemini, but .env still contains OPENAI_API_KEY. "
                "Open Google AI Studio to create a Gemini API key, then replace the "
                "OPENAI_API_KEY line in .env with GEMINI_API_KEY=<your-gemini-key>. "
                "Keep PINECONE_API_KEY. Do not use an OpenAI key as a Gemini key."
            )
        raise RuntimeError(
            f"Missing environment variables: {', '.join(missing)}. "
            "Set GEMINI_API_KEY (or GOOGLE_API_KEY) and PINECONE_API_KEY in .env."
        )
