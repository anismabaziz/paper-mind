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

PROVIDER_API_KEYS = {"google": "GOOGLE_API_KEY", "groq": "GROQ_API_KEY"}


def _vector_backend():
    return os.getenv("VECTOR_BACKEND", "qdrant").lower()


def missing_required_vars():
    """
        Return the list of required environment variables that are unset or invalid.
    """
    # Read MODE fresh from the environment so validation reflects the
    # process's current state, not whatever was set at import time.
    mode = os.getenv("MODE", "google").lower()
    vector_backend = _vector_backend()
    required = ["DATABASE_URL"]
    if vector_backend == "pinecone":
        required.append("PINECONE_API_KEY")
    elif vector_backend == "qdrant":
        pass
    else:
        # Invalid backend will be reported below; don't also demand Pinecone
        required.append("VECTOR_BACKEND")

    missing = [var for var in required if not os.getenv(var)]

    if vector_backend not in ("qdrant", "pinecone"):
        missing.append(f"VECTOR_BACKEND (got {vector_backend!r}, expected 'qdrant' or 'pinecone')")

    provider_key = PROVIDER_API_KEYS.get(mode)
    if provider_key is None:
        missing.append(f"MODE (got {mode!r}, expected 'google' or 'groq')")
    elif not os.getenv(provider_key):
        missing.append(provider_key)

    return missing


def validate():
    """
        Exit with a readable error if any required variable is missing.
    """
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


def get_vector_index():
    global _pinecone_index
    # Tests monkeypatch ``_pinecone_index`` directly with a fake. Honor that
    # fake regardless of VECTOR_BACKEND so existing tests keep working when
    # the default flips to qdrant.
    vector_backend = _vector_backend()
    if vector_backend == "qdrant":
        if _qdrant_index is not None:
            return _qdrant_index
        if _pinecone_index is not None:
            # Legacy fake installed via ``_pinecone_index`` – keep using it
            # for older tests that only patch the Pinecone seam.
            return _pinecone_index
        return _get_qdrant_index()

    # pinecone path
    if _pinecone_index is None:
        from pinecone import Pinecone

        _pinecone_index = Pinecone(api_key=os.getenv("PINECONE_API_KEY")).Index(
            INDEX_NAME
        )
    return _pinecone_index


def get_genai_client():
    global _genai_client
    if _genai_client is None:
        from google import genai

        _genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _genai_client


def get_groq_client():
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
