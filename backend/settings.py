"""
Typed, validated application settings.

A single pydantic-settings object reads every environment variable the
backend consumes, grouped by concern. Importing this module never builds a
client or touches an external service — clients live in ``providers`` and
are built lazily on first use.

Field names mirror their environment variable (``chunk_size_tokens`` ←
``CHUNK_SIZE_TOKENS``) so validation failures name the variable to set.

This module sits beside the legacy ``config`` module until every call site
is migrated; the old module stays authoritative in the meantime.
"""

import sys
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Backend directory (storage paths are rooted here)
BACKEND_DIR = Path(__file__).resolve().parent

_TRUE = ("1", "true", "yes")


def _parse_bool(value):
    """Accept the truthy spellings the legacy config module allowed."""
    if isinstance(value, str):
        return value.lower() in _TRUE
    return value


class DatabaseSettings(BaseSettings):
    """Relational database (Postgres in dev via compose; required)."""

    model_config = SettingsConfigDict(extra="ignore")

    database_url: str = Field(validation_alias="DATABASE_URL")

    @field_validator("database_url", mode="before")
    @classmethod
    def _non_empty(cls, value):
        if value is None or not str(value).strip():
            raise ValueError("must be set (see backend/.env.example)")
        return value


class StorageSettings(BaseSettings):
    """Where uploaded PDFs live on disk."""

    model_config = SettingsConfigDict(extra="ignore")

    storage_dir: Path = Field(
        default=BACKEND_DIR / "data" / "storage", validation_alias="STORAGE_DIR"
    )


class VectorSettings(BaseSettings):
    """Qdrant connection and collection."""

    model_config = SettingsConfigDict(extra="ignore")

    qdrant_url: str = Field(
        default="http://localhost:6333", validation_alias="QDRANT_URL"
    )
    index_name: str = "pdf-index"


class EmbeddingSettings(BaseSettings):
    """
    Local embedding model (free, CPU-capable).

    BGE-M3 supports dense+sparse, 8192 context, and Matryoshka truncation to 1024d.
    """

    model_config = SettingsConfigDict(extra="ignore")

    embedding_model: str = Field(
        default="BAAI/bge-m3", validation_alias="LOCAL_EMBEDDING_MODEL"
    )


class ChunkingSettings(BaseSettings):
    """
    Token-based chunking via tiktoken cl100k_base.

    512 tokens ~2000 chars, 10% overlap ~50.
    """

    model_config = SettingsConfigDict(extra="ignore")

    chunk_size_tokens: int = Field(default=512, validation_alias="CHUNK_SIZE_TOKENS")
    chunk_overlap_tokens: int = Field(
        default=50, validation_alias="CHUNK_OVERLAP_TOKENS"
    )


class RerankSettings(BaseSettings):
    """
    Gated local cross-encoder over hybrid candidates.

    Alternative quality model: BAAI/bge-reranker-v2-m3
    (~80ms/50 docs vs ~10ms/50 for MiniLM).
    """

    model_config = SettingsConfigDict(extra="ignore")

    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        validation_alias="RERANK_MODEL",
    )
    enabled: bool = Field(default=False, validation_alias="RERANK")

    @field_validator("enabled", mode="before")
    @classmethod
    def _coerce(cls, value):
        return _parse_bool(value)


class ParsingSettings(BaseSettings):
    """
    Docling branch routing (auto | true | false).

    - auto: lightweight heuristic routes only image-only / borderless-table /
      2-col PDFs to Docling; others stay on pymupdf fast path.
    - true: force Docling for all PDFs (requires the `docling` extra).
    - false: never use Docling, always pymupdf.
    """

    model_config = SettingsConfigDict(extra="ignore")

    use_docling: str = Field(default="auto", validation_alias="USE_DOCLING")

    @field_validator("use_docling")
    @classmethod
    def _known_mode(cls, value):
        if str(value).lower() not in ("auto", "true", "false"):
            raise ValueError("USE_DOCLING must be one of: auto, true, false")
        return str(value).lower()


class AuthSettings(BaseSettings):
    """
    JWT signing and demo mode.

    The JWT secret also derives the Fernet key that encrypts stored user
    API keys, so changing it invalidates those keys.
    """

    model_config = SettingsConfigDict(extra="ignore")

    jwt_secret: str | None = Field(default=None, validation_alias="JWT_SECRET")
    demo_mode: bool = Field(default=False, validation_alias="DEMO_MODE")

    @field_validator("demo_mode", mode="before")
    @classmethod
    def _coerce(cls, value):
        return _parse_bool(value)


class Settings(BaseSettings):
    """Every environment variable the backend consumes, validated at build."""

    model_config = SettingsConfigDict(extra="ignore")

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    vector: VectorSettings = Field(default_factory=VectorSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    # The alias keeps the env source from matching RERANK here — that
    # variable belongs to the nested group's boolean, not to the group.
    rerank: RerankSettings = Field(
        default_factory=RerankSettings, validation_alias=AliasChoices("rerank_group")
    )
    parsing: ParsingSettings = Field(default_factory=ParsingSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build the process-wide settings instance once."""
    return Settings()


def jwt_secret_warning(settings: Settings) -> str | None:
    """Return the warning message when the JWT secret is unset outside demo mode."""
    if settings.auth.demo_mode or settings.auth.jwt_secret:
        return None
    return (
        "Warning: DEMO_MODE is off but JWT_SECRET is unset; issued tokens "
        "will stop working after a restart. Set JWT_SECRET in .env."
    )


def validate(settings: Settings | None = None) -> None:
    """Exit with a readable error if required configuration is missing."""
    try:
        settings = settings or get_settings()
    except ValidationError as e:
        print("PaperMind backend is missing required configuration:", file=sys.stderr)
        for error in e.errors():
            env_var = str(error["loc"][-1]).upper()
            print(f"  - {env_var}: {error['msg']}", file=sys.stderr)
        print(
            "\nFix: copy backend/.env.example to backend/.env and fill in the "
            "values above, then start the app again.",
            file=sys.stderr,
        )
        sys.exit(1)

    warning = jwt_secret_warning(settings)
    if warning:
        print(warning, file=sys.stderr)
