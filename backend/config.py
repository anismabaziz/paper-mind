"""
Application configuration.

Environment variables are validated lazily: importing this module never
builds a client or touches an external service. Clients are created on
first access through module-level ``__getattr__`` so routes only pay for
what they actually use.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Backend directory (storage paths are rooted here)
BACKEND_DIR = Path(__file__).resolve().parent

# Where uploaded PDFs live on disk
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", str(BACKEND_DIR / "data" / "storage")))

# Relational database (Postgres in dev via compose; validated at boot)
DATABASE_URL = os.getenv("DATABASE_URL")

# App Constants
INDEX_NAME = "pdf-index"

# Mode Selector: which provider answers chat requests ("google" or "groq")
MODE = os.getenv("MODE", "google").lower()

# Demo mode (DEMO_MODE env) is read per-request by the auth service, so it
# can be toggled without reloading this module.

# Model Constants
EMBEDDING_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.0-flash"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Local embedding model (free, CPU-capable). BGE-M3 supports dense+sparse, 8192
# context, and Matryoshka truncation to 1024d.
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-m3")
LOCAL_EMBED_DIM = int(os.getenv("LOCAL_EMBED_DIM", "1024"))

# Hybrid retrieval: dense + BM25 sparse fusion
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.7"))
FETCH_K = int(os.getenv("FETCH_K", "50"))
RRF_K = int(os.getenv("RRF_K", "60"))

# Chunking: token-based via tiktoken cl100k_base (512 tokens ~2000 chars, 10% overlap ~50)
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "512"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))

# Reranker: gated local cross-encoder over FETCH_K candidates
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# Alternative quality model: BAAI/bge-reranker-v2-m3 (~80ms/50 docs vs ~10ms/50 for MiniLM)


def is_rerank_enabled() -> bool:
    """Do is rerank enabled."""
    return os.getenv("RERANK", "false").lower() in ("1", "true", "yes")


PROVIDER_API_KEYS = {"google": "GOOGLE_API_KEY", "groq": "GROQ_API_KEY"}

from enum import Enum


class VectorBackend(str, Enum):
    """VectorBackend."""

    QDRANT = "qdrant"
    PINECONE = "pinecone"


class EmbedBackend(str, Enum):
    """EmbedBackend."""

    LOCAL = "local"
    GEMINI = "gemini"


class ChatProvider(str, Enum):
    """ChatProvider."""

    GOOGLE = "google"
    GROQ = "groq"


_VALID_VECTOR_BACKENDS = {b.value for b in VectorBackend}
_VALID_EMBED_BACKENDS = {b.value for b in EmbedBackend}
_VALID_PROVIDERS = {p.value for p in ChatProvider}


def _vector_backend() -> str:
    return os.getenv("VECTOR_BACKEND", VectorBackend.QDRANT.value).lower()


def _embed_backend() -> str:
    return os.getenv("EMBED_BACKEND", EmbedBackend.LOCAL.value).lower()


def _chat_provider() -> str:
    return os.getenv("MODE", ChatProvider.GOOGLE.value).lower()


def missing_required_vars():
    """
    Return the list of required environment variables that are unset or invalid.

    Validation is driven by the three backend enums above so adding a new
    backend only touches the enum definition, not a cascade of if/else.
    """
    mode = _chat_provider()
    vector_backend = _vector_backend()
    embed_backend = _embed_backend()

    required = ["DATABASE_URL"]
    # Vector backend requirement is a single map lookup, not a cascade
    if vector_backend == VectorBackend.PINECONE.value:
        required.append("PINECONE_API_KEY")
    elif vector_backend not in _VALID_VECTOR_BACKENDS:
        required.append("VECTOR_BACKEND")

    missing = [var for var in required if not os.getenv(var)]

    if vector_backend not in _VALID_VECTOR_BACKENDS:
        missing.append(
            f"VECTOR_BACKEND (got {vector_backend!r}, expected 'qdrant' or 'pinecone')"
        )

    provider_key = PROVIDER_API_KEYS.get(mode)
    if mode not in _VALID_PROVIDERS:
        missing.append(f"MODE (got {mode!r}, expected 'google' or 'groq')")
    elif provider_key and not os.getenv(provider_key):
        missing.append(provider_key)

    if embed_backend not in _VALID_EMBED_BACKENDS:
        missing.append(
            f"EMBED_BACKEND (got {embed_backend!r}, expected 'local' or 'gemini')"
        )
    elif embed_backend == EmbedBackend.GEMINI.value and not os.getenv("GOOGLE_API_KEY"):
        if "GOOGLE_API_KEY" not in missing:
            missing.append("GOOGLE_API_KEY")

    return missing


def validate():
    """Exit with a readable error if any required variable is missing."""
    missing = missing_required_vars()
    if missing:
        print("PaperMind backend is missing required configuration:", file=sys.stderr)
        for var in missing:
            print(f"  - {var}", file=sys.stderr)
        print(
            "\nFix: copy backend/.env.example to backend/.env and fill in the "
            "values above, then start the app again.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.getenv("DEMO_MODE", "").lower() in ("1", "true", "yes") and not os.getenv(
        "JWT_SECRET"
    ):
        print(
            "Warning: DEMO_MODE is off but JWT_SECRET is unset; issued tokens "
            "will stop working after a restart. Set JWT_SECRET in .env.",
            file=sys.stderr,
        )


# --- Lazy clients -----------------------------------------------------------
#
# Nothing below this line runs at import time. Each client is built once, on
# first access, so a misconfigured key fails at the moment of use with a
# clear origin instead of crashing the whole import.

_pinecone_index = None
_qdrant_client = None
_qdrant_index = None
_genai_client = None
_groq_client = None


def _get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient

        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        _qdrant_client = QdrantClient(url=url)
    return _qdrant_client


def _get_qdrant_index():
    global _qdrant_index
    if _qdrant_index is None:
        from services.qdrant_store import QdrantIndexAdapter

        _qdrant_index = QdrantIndexAdapter(_get_qdrant_client(), INDEX_NAME)
    return _qdrant_index


def _get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        from pinecone import Pinecone

        _pinecone_index = Pinecone(api_key=os.getenv("PINECONE_API_KEY")).Index(
            INDEX_NAME
        )
    return _pinecone_index


_VECTOR_FACTORIES = {
    VectorBackend.QDRANT.value: _get_qdrant_index,
    VectorBackend.PINECONE.value: _get_pinecone_index,
}


def get_vector_index():
    # Tests monkeypatch ``_pinecone_index`` directly with a fake. Honor that
    # fake regardless of VECTOR_BACKEND so existing tests keep working when
    # the default flips to qdrant (dispatch map keeps the cascade in one place).
    """Do get vector index."""
    vector_backend = _vector_backend()
    if vector_backend == VectorBackend.QDRANT.value:
        if _qdrant_index is not None:
            return _qdrant_index
        if _pinecone_index is not None:
            return _pinecone_index
        return _get_qdrant_index()

    factory = _VECTOR_FACTORIES.get(vector_backend)
    if factory is not None:
        return factory()
    # Invalid backend – let missing_required_vars report it; fall back to qdrant for boot
    return _get_qdrant_index()


def get_genai_client():
    """Do get genai client."""
    global _genai_client
    if _genai_client is None:
        from google import genai

        _genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _genai_client


def get_groq_client():
    """Do get groq client."""
    global _groq_client
    if _groq_client is None:
        from groq import Groq

        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


_LAZY_ATTRS = {
    "vector_index": get_vector_index,
    "genai_client": get_genai_client,
    "groq_client": get_groq_client,
}


def __getattr__(name):
    builder = _LAZY_ATTRS.get(name)
    if builder is None:
        raise AttributeError(f"module 'config' has no attribute {name!r}")
    return builder()


def __dir__():
    return sorted(list(globals()) + list(_LAZY_ATTRS))
