
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _get_int(name: str, default: int) -> int:
    """Read an integer setting from the environment."""
    raw_value = os.getenv(name)

    if raw_value is None or raw_value.strip() == "":
        return default

    try:
        return int(raw_value)
    except ValueError:
        return default


# Folders
PROJECT_ROOT = Path(__file__).parent
DOCUMENT_DIR = PROJECT_ROOT / "data" / "documents"
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

FAISS_DIR = VECTORSTORE_DIR / "faiss"
DOC_STORE_PATH = VECTORSTORE_DIR / "doc_store.json"


# Models
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_CHAT_MODEL = os.getenv(
    "GROQ_CHAT_MODEL", "openai/gpt-oss-120b"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

LLM_TEMPERATURE = 0.0


# Vector database
VECTOR_DB = os.getenv("VECTOR_DB", "faiss").strip().lower()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME", "advance-rag"
)
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")


# Chunking
PARENT_CHUNK_SIZE = _get_int("PARENT_CHUNK_SIZE", 1600)
PARENT_CHUNK_OVERLAP = _get_int("PARENT_CHUNK_OVERLAP", 200)

CHILD_CHUNK_SIZE = _get_int("CHILD_CHUNK_SIZE", 400)
CHILD_CHUNK_OVERLAP = _get_int("CHILD_CHUNK_OVERLAP", 60)


# Query expansion
NUM_QUERY_VARIANTS = _get_int("NUM_QUERY_VARIANTS", 3)


# Retrieval
VECTOR_TOP_K = _get_int("VECTOR_TOP_K", 6)
BM25_TOP_K = _get_int("BM25_TOP_K", 6)
HYBRID_TOP_K = _get_int("HYBRID_TOP_K", 10)

RERANK_TOP_K = _get_int("RERANK_TOP_K", 4)
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "-4.0"))

MAX_CONTEXT_CHARS = _get_int("MAX_CONTEXT_CHARS", 8000)

RERANKER = os.getenv("RERANKER", "cross-encoder").strip().lower()

CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# Conversation memory
MEMORY_WINDOW = _get_int("MEMORY_WINDOW", 4)


def check_api_key() -> None:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set")


def check_pinecone_config() -> None:
    if VECTOR_DB == "pinecone":
        if not PINECONE_API_KEY:
            raise ValueError(
                "VECTOR_DB=pinecone but PINECONE_API_KEY is not set"
            )

        if not PINECONE_INDEX_NAME:
            raise ValueError(
                "VECTOR_DB=pinecone but PINECONE_INDEX_NAME is not set"
            )


def describe_settings() -> dict:
    return {
        "Chat model": GROQ_CHAT_MODEL,
        "Embedding model": EMBEDDING_MODEL,
        "Vector DB": VECTOR_DB,
        "Parent chunk size": PARENT_CHUNK_SIZE,
        "Child chunk size": CHILD_CHUNK_SIZE,
        "Reranker": RERANKER,
        "Cross-encoder model": CROSS_ENCODER_MODEL,
        "Memory window": MEMORY_WINDOW,
    }
