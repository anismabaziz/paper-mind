"""
Lazy singleton clients for external services.

Nothing here runs at import time: each client is built once, on first
access, so a misconfigured service fails at the moment of use with a clear
origin instead of crashing the whole import. Configuration comes from the
typed ``Settings`` object, never from the environment directly.
"""

import threading

from settings import get_settings

_qdrant_client = None
_qdrant_index = None
_qdrant_lock = threading.RLock()
_qdrant_index_lock = threading.RLock()


def build_qdrant_client(
    qdrant_url: str, qdrant_api_key: str | None = None, timeout: int | None = None
):
    """Build a Qdrant client from an explicit URL and optional API key."""
    from qdrant_client import QdrantClient

    kwargs: dict = {"url": qdrant_url, "api_key": qdrant_api_key or None}
    if timeout is not None:
        kwargs["timeout"] = timeout
    return QdrantClient(**kwargs)


def get_qdrant_client():
    """Do get qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        with _qdrant_lock:
            if _qdrant_client is None:
                vector = get_settings().vector
                _qdrant_client = build_qdrant_client(
                    vector.qdrant_url, vector.qdrant_api_key
                )
    return _qdrant_client


def get_vector_index():
    """Do get vector index."""
    global _qdrant_index
    if _qdrant_index is None:
        with _qdrant_index_lock:
            if _qdrant_index is None:
                from services.retrieval.qdrant_store import QdrantIndexAdapter

                _qdrant_index = QdrantIndexAdapter(
                    get_qdrant_client(), get_settings().vector.index_name
                )
    return _qdrant_index
