"""
Index Manifest: what a Document's vectors were built with.

Every indexed Document records the manifest that produced its active Index
Generation: parser, chunking, embedding, sparse retrieval, reranking, and
collection schema. When the running application no longer matches that
manifest the Document is stale — its vectors describe a configuration the app
no longer serves, so chat must stop and ask for a reindex instead of querying
an incompatible index.

The runtime manifest is derived from :class:`settings.Settings`, so a
configuration or model change is visible without any extra bookkeeping. The
content hash and Index Generation are recorded rather than compared: they
describe the stored index, not the running configuration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from typing import Any

from services.parsing.document_parser import PARSER_VERSION
from services.retrieval.hybrid import SPARSE_METHOD, TOKENIZER_VERSION

# Bumped whenever the Qdrant collection shape changes in a way that makes
# previously stored points unusable.
COLLECTION_SCHEMA_VERSION = "named-dense-sparse-v1"

# Fields the running configuration owns. A difference in any of them makes a
# stored index incompatible with the running application. The reranker gate is
# recorded but not compared: it decides at query time whether reranking runs,
# so toggling it changes what is served without changing what was stored.
RUNTIME_FIELDS: tuple[str, ...] = (
    "parser",
    "parser_version",
    "chunk_size_tokens",
    "chunk_overlap_tokens",
    "embedding_model",
    "embedding_revision",
    "vector_dimension",
    "sparse_method",
    "sparse_tokenizer_version",
    "reranker_model",
    "reranker_revision",
    "collection_name",
    "collection_schema_version",
)

# Reported instead of a field name when a processed document predates
# manifest recording, so its provenance is unknown.
MISSING_MANIFEST_CHANGE = "index_manifest"

# User-facing names for the settings that force a reindex.
CHANGE_LABELS: dict[str, str] = {
    "parser": "parser",
    "parser_version": "parser version",
    "chunk_size_tokens": "chunk size",
    "chunk_overlap_tokens": "chunk overlap",
    "embedding_model": "embedding model",
    "embedding_revision": "embedding revision",
    "vector_dimension": "vector dimension",
    "sparse_method": "sparse retrieval method",
    "sparse_tokenizer_version": "sparse tokenizer version",
    "reranker_model": "reranker model",
    "reranker_revision": "reranker revision",
    "reranker_enabled": "reranker setting",
    "collection_name": "vector collection",
    "collection_schema_version": "collection schema version",
    MISSING_MANIFEST_CHANGE: "index manifest",
}


def content_hash(file_bytes: bytes) -> str:
    """Return the SHA-256 of the document bytes that were indexed."""
    return hashlib.sha256(file_bytes).hexdigest()


@dataclass(frozen=True)
class IndexManifest:
    """Everything that decides what a Document's vectors mean."""

    content_hash: str
    # The Docling routing mode, not the branch one document happened to take:
    # the parser that ran is decided per document by the layout heuristics,
    # and a change to those heuristics is covered by parser_version.
    parser: str
    parser_version: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    embedding_model: str
    embedding_revision: str
    vector_dimension: int
    sparse_method: str
    sparse_tokenizer_version: str
    reranker_model: str
    reranker_revision: str
    reranker_enabled: bool
    collection_name: str
    collection_schema_version: str
    index_generation: int

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest as a JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Return the manifest as stored JSON with stable key order."""
        return json.dumps(self.to_dict(), sort_keys=True)


def vector_dimension() -> int:
    """Return the dense vector width the collection is created with."""
    from services.retrieval.qdrant_store import DENSE_SIZE

    return DENSE_SIZE


def runtime_manifest(
    settings: Any,
    *,
    content_hash_value: str = "",
    index_generation: int = 0,
) -> IndexManifest:
    """Build the manifest the running configuration would produce."""
    return IndexManifest(
        content_hash=content_hash_value,
        parser=settings.parsing.use_docling,
        parser_version=PARSER_VERSION,
        chunk_size_tokens=settings.chunking.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunking.chunk_overlap_tokens,
        embedding_model=settings.embedding.embedding_model,
        embedding_revision=settings.embedding.revision,
        vector_dimension=vector_dimension(),
        sparse_method=SPARSE_METHOD,
        sparse_tokenizer_version=TOKENIZER_VERSION,
        reranker_model=settings.rerank.rerank_model,
        reranker_revision=settings.rerank.revision,
        reranker_enabled=settings.rerank.enabled,
        collection_name=settings.vector.index_name,
        collection_schema_version=COLLECTION_SCHEMA_VERSION,
        index_generation=index_generation,
    )


def manifest_builder(settings: Any):
    """Return a builder that records this configuration's manifest."""

    def build(file_bytes: bytes, index_generation: int) -> IndexManifest:
        return runtime_manifest(
            settings,
            content_hash_value=content_hash(file_bytes),
            index_generation=index_generation,
        )

    return build


def manifest_from_json(value: str | None) -> IndexManifest | None:
    """Rebuild a stored manifest, or None when it is missing or unreadable."""
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    known = {field.name for field in fields(IndexManifest)}
    if not known.issubset(parsed.keys()):
        return None
    try:
        return IndexManifest(**parsed)
    except TypeError:
        return None


def manifest_changes(stored: IndexManifest | None, current: IndexManifest) -> list[str]:
    """Return the runtime fields a stored manifest no longer matches."""
    if stored is None:
        return []
    return [
        name
        for name in RUNTIME_FIELDS
        if getattr(stored, name) != getattr(current, name)
    ]


def change_labels(changes: list[str]) -> list[str]:
    """Return user-facing names for the settings that force a reindex."""
    return [CHANGE_LABELS.get(name, name) for name in changes]


def change_details(
    stored: IndexManifest | None, changes: list[str], settings: Any
) -> list[dict[str, Any]]:
    """Return each changed setting with the value it was built with and now has."""
    current = runtime_manifest(settings)
    return [
        {
            "field": name,
            "label": CHANGE_LABELS.get(name, name),
            "indexed": getattr(stored, name, None) if stored else None,
            "current": getattr(current, name, None),
        }
        for name in changes
    ]
