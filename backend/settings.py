"""
Typed, validated application settings.

A single pydantic-settings object reads every environment variable the
backend consumes, grouped by concern. Application modules consume a
``Settings`` instance rather than reading the environment themselves.
Alembic reads ``DATABASE_URL`` directly so database upgrades do not import
application code. Importing this module never builds a client or touches an
external service. Clients live in ``providers`` and are built lazily on first
use.

Field names generally mirror their environment variable
(``chunk_size_tokens`` ← ``CHUNK_SIZE_TOKENS``) so validation failures
name the variable to set.
"""

import sys
import threading
from pathlib import Path

from dotenv import load_dotenv
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Local dev boots via `python app.py` with values in backend/.env; existing
# process env vars keep precedence (dotenv never overrides them).
load_dotenv()

# Backend directory (storage paths are rooted here)
BACKEND_DIR = Path(__file__).resolve().parent

_TRUE = ("1", "true", "yes")


def _parse_bool(value):
    """Accept the truthy spellings the legacy config module allowed."""
    if isinstance(value, str):
        return value.lower() in _TRUE
    return value


class DatabaseSettings(BaseSettings):
    """
    Relational database (Postgres in dev via compose; required at boot).

    The URL defaults to empty so building a Settings never raises — imports
    stay side-effect free (as under the legacy config module) and
    :func:`validate` reports the missing variable with a readable error.
    """

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    database_url: str = Field(default="", validation_alias="DATABASE_URL")


class StorageSettings(BaseSettings):
    """Where uploaded PDFs live on disk."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    storage_dir: Path = Field(
        default=BACKEND_DIR / "data" / "storage", validation_alias="STORAGE_DIR"
    )


class VectorSettings(BaseSettings):
    """Qdrant connection and collection."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    qdrant_url: str = Field(
        default="http://localhost:6333", validation_alias="QDRANT_URL"
    )
    index_name: str = "pdf-index"


class EmbeddingSettings(BaseSettings):
    """
    Local embedding model (free, CPU-capable).

    BGE-M3 supports dense+sparse, 8192 context, and Matryoshka truncation to 1024d.
    """

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    embedding_model: str = Field(
        default="BAAI/bge-m3", validation_alias="LOCAL_EMBEDDING_MODEL"
    )


class ChunkingSettings(BaseSettings):
    """
    Token-based chunking via tiktoken cl100k_base.

    512 tokens ~2000 chars, 10% overlap ~50.
    """

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

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

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

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

    Any other spelling falls through to the heuristic, as before.
    """

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    use_docling: str = Field(default="auto", validation_alias="USE_DOCLING")


class AuthSettings(BaseSettings):
    """
    App secret for encrypting stored provider API keys.

    The Fernet key is derived from ``APP_SECRET`` via HKDF-SHA256 with a
    versioned info string (urlsafe base64). Changing it invalidates
    previously encrypted keys.
    """

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    app_secret: str | None = Field(default=None, validation_alias="APP_SECRET")


class UploadSettings(BaseSettings):
    """Upload limits and allowed content types."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    max_upload_bytes: int = Field(
        default=50 * 1024 * 1024,
        validation_alias=AliasChoices("MAX_UPLOAD_BYTES", "MAX_CONTENT_LENGTH"),
    )
    allowed_extensions: set[str] = {".pdf"}
    allowed_mime_types: set[str] = {"application/pdf"}


class FrontendSettings(BaseSettings):
    """Frontend origin for CORS allowlist."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    frontend_origin: str = Field(
        default="http://localhost:5173",
        validation_alias=AliasChoices("FRONTEND_ORIGIN", "FRONTEND_URL", "CORS_ALLOWED_ORIGINS"),
    )


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
    upload: UploadSettings = Field(default_factory=UploadSettings)
    frontend: FrontendSettings = Field(default_factory=FrontendSettings)


_settings: Settings | None = None
_settings_lock = threading.RLock()


def get_settings() -> Settings:
    """Return the process-wide settings instance, building it once."""
    global _settings
    if _settings is None:
        with _settings_lock:
            if _settings is None:
                _settings = Settings()
    return _settings


def set_settings(settings: Settings | None) -> None:
    """Install or clear the process-wide instance (app boot and tests)."""
    global _settings
    with _settings_lock:
        _settings = settings


def app_secret_warning(settings: Settings) -> str | None:
    """Return a warning when APP_SECRET is unset (keys need it to decrypt)."""
    if settings.auth.app_secret:
        return None
    return (
        "Warning: APP_SECRET is unset; stored provider keys cannot be "
        "encrypted or decrypted without it. Set APP_SECRET in .env."
    )


# Backwards compatibility: older imports reference jwt_secret_warning.
def jwt_secret_warning(settings: Settings) -> str | None:
    """Legacy alias for app_secret_warning."""
    return app_secret_warning(settings)


def validate(settings: Settings | None = None) -> None:
    """Exit with a readable error if required configuration is missing."""
    settings = settings or get_settings()
    missing = []
    if not settings.database.database_url.strip():
        missing.append("DATABASE_URL")
    if missing:
        print("PaperMind backend is missing required configuration:", file=sys.stderr)
        for var in missing:
            print(f"  - {var}: must be set (see backend/.env.example)", file=sys.stderr)
        print(
            "\nFix: copy backend/.env.example to backend/.env and fill in the "
            "values above, then start the app again.",
            file=sys.stderr,
        )
        sys.exit(1)

    warning = app_secret_warning(settings)
    if warning:
        print(warning, file=sys.stderr)
