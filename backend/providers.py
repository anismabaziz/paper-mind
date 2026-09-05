"""
Lazy singleton clients for external services.

Nothing here runs at import time: each client is built once, on first
access, so a misconfigured service fails at the moment of use with a clear
origin instead of crashing the whole import. Configuration comes from the
typed ``Settings`` object, never from the environment directly.
"""

from settings import get_settings

_qdrant_client = None
_qdrant_index = None


def get_qdrant_client():
    """Do get qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient

        _qdrant_client = QdrantClient(url=get_settings().vector.qdrant_url)
    return _qdrant_client


def get_vector_index():
    """Do get vector index."""
    global _qdrant_index
    if _qdrant_index is None:
        from services.qdrant_store import QdrantIndexAdapter

        _qdrant_index = QdrantIndexAdapter(
            get_qdrant_client(), get_settings().vector.index_name
        )
    return _qdrant_index


def reset_providers() -> None:
    """Drop memoized clients (tests install fakes or rewire between runs)."""
    global _qdrant_client, _qdrant_index
    _qdrant_client = None
    _qdrant_index = None
