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
import urllib.parse
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
    # Optional API key sent to Qdrant. Empty for loopback development;
    # required before any non-loopback Qdrant address is accepted.
    qdrant_api_key: str | None = Field(default=None, validation_alias="QDRANT_API_KEY")
    index_name: str = "pdf-index"


#: Hosts that keep traffic on the local machine. Anything else — including
#: "0.0.0.0", "::", a LAN address, a compose service name, or an
#: unparseable value — is treated as remote and fails closed.
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def url_is_loopback(url: str) -> bool:
    """Return True only when a Qdrant URL clearly targets this machine."""
    try:
        host = urllib.parse.urlsplit(url.strip()).hostname
    except ValueError:
        return False
    if not host:
        return False
    return host.lower() in LOOPBACK_HOSTS


class EmbeddingSettings(BaseSettings):
    """
    Local embedding model (free, CPU-capable).

    BGE-M3 supports dense+sparse, 8192 context, and Matryoshka truncation to 1024d.
    """

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    embedding_model: str = Field(
        default="BAAI/bge-m3", validation_alias="LOCAL_EMBEDDING_MODEL"
    )
    # Pinned immutable revision (a commit sha) of the embedding model. Changing
    # it makes every stored vector incompatible and marks documents stale.
    # Empty means the commit recorded in ``services.models`` for this
    # repository is loaded, which is every shipped default.
    revision: str = Field(default="", validation_alias="LOCAL_EMBEDDING_REVISION")
    # Whether the model may execute Python from its own repository. Off by
    # default: BAAI/bge-m3 loads through stock transformers classes, so turning
    # it on adds risk without adding capability. Only a commit recorded in
    # ``services.models.REVIEWED_REMOTE_CODE_MODELS`` is accepted, so the
    # question is never whether to trust a moving branch.
    trust_remote_code: bool = Field(
        default=False, validation_alias="LOCAL_EMBEDDING_TRUST_REMOTE_CODE"
    )

    @field_validator("trust_remote_code", mode="before")
    @classmethod
    def _coerce_remote_code(cls, value):
        return _parse_bool(value)


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
    # Pinned immutable revision (a commit sha) of the reranker model. Changing
    # it makes every stored rerank order incompatible and marks documents stale.
    # Empty means the commit recorded in ``services.models`` for this
    # repository is loaded, which is every shipped default.
    revision: str = Field(default="", validation_alias="RERANK_REVISION")
    # See ``EmbeddingSettings.trust_remote_code``; the reasoning is identical.
    trust_remote_code: bool = Field(
        default=False, validation_alias="RERANK_TRUST_REMOTE_CODE"
    )

    @field_validator("enabled", "trust_remote_code", mode="before")
    @classmethod
    def _coerce(cls, value):
        return _parse_bool(value)


class QueryContextSettings(BaseSettings):
    """
    Bounds on the context one chat request carries.

    The Conversation window is bounded by both ``chat_recent_turns`` and
    ``chat_prior_turns_token_budget``; retrieved evidence by
    ``chat_context_token_budget``. Query rewriting spends an extra model call,
    so it stays off until the evaluation set shows it earns its cost.
    """

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    recent_turns: int = Field(default=4, ge=0, validation_alias="CHAT_RECENT_TURNS")
    prior_turns_token_budget: int = Field(
        default=1200, ge=0, validation_alias="CHAT_PRIOR_TURNS_TOKEN_BUDGET"
    )
    context_token_budget: int = Field(
        default=6000, ge=0, validation_alias="CHAT_CONTEXT_TOKEN_BUDGET"
    )
    query_rewrite: bool = Field(default=False, validation_alias="CHAT_QUERY_REWRITE")
    max_expansion_chars: int = Field(
        default=2000, ge=0, validation_alias="CHAT_MAX_EXPANSION_CHARS"
    )

    @field_validator("query_rewrite", mode="before")
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


DEFAULT_MAX_INGESTION_PAGES = 1000
DEFAULT_MAX_INGESTION_TEXT_BYTES = 100 * 1024 * 1024
DEFAULT_MAX_INGESTION_OUTPUT_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_INGESTION_SECONDS = 1800.0
DEFAULT_MAX_INGESTION_MEMORY_BYTES = 4 * 1024 * 1024 * 1024


class IngestionSettings(BaseSettings):
    """Resource limits applied to each durable ingestion job."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    max_pages: int = Field(
        default=DEFAULT_MAX_INGESTION_PAGES,
        ge=0,
        validation_alias="MAX_INGESTION_PAGES",
    )
    max_extracted_text_bytes: int = Field(
        default=DEFAULT_MAX_INGESTION_TEXT_BYTES,
        ge=0,
        validation_alias="MAX_INGESTION_TEXT_BYTES",
    )
    max_output_bytes: int = Field(
        default=DEFAULT_MAX_INGESTION_OUTPUT_BYTES,
        ge=0,
        validation_alias="MAX_INGESTION_OUTPUT_BYTES",
    )
    max_elapsed_seconds: float = Field(
        default=DEFAULT_MAX_INGESTION_SECONDS,
        ge=0,
        validation_alias="MAX_INGESTION_SECONDS",
    )
    max_memory_bytes: int = Field(
        default=DEFAULT_MAX_INGESTION_MEMORY_BYTES,
        ge=0,
        validation_alias="MAX_INGESTION_MEMORY_BYTES",
    )


class ResearchSettings(BaseSettings):
    """
    Spend ceilings for one Research Brief.

    A brief lets a model choose what to read next, which is the property that
    makes it able to spend without being asked. These are the bounds that keep
    one question from becoming an unbounded bill, and they are per brief rather
    than per session so a reader sees the same ceiling whatever they are doing.
    """

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    #: How many times the model may think before a brief is stopped.
    max_turns: int = Field(default=6, ge=1, validation_alias="RESEARCH_MAX_TURNS")
    #: How many tool calls one brief may make, which is what bounds retrieval.
    max_tool_calls: int = Field(
        default=8, ge=1, validation_alias="RESEARCH_MAX_TOOL_CALLS"
    )
    #: How many times one identical call may be made. A model stuck in a loop
    #: re-asks rather than alternating, so this is what catches that.
    max_repeated_calls: int = Field(
        default=2, ge=1, validation_alias="RESEARCH_MAX_REPEATED_CALLS"
    )
    #: What one brief may bill across every turn.
    max_tokens: int = Field(
        default=120_000, ge=1, validation_alias="RESEARCH_MAX_TOKENS"
    )
    #: What a reader waits before a brief is stopped between turns.
    max_seconds: float = Field(
        default=120.0, ge=1.0, validation_alias="RESEARCH_MAX_SECONDS"
    )


class TelemetrySettings(BaseSettings):
    """
    Traces for the answer path: where they go, and what they may contain.

    Local export is on by default and writes redacted spans to a file the
    operator can read while the application runs. Nothing is sent anywhere
    unless ``otlp_endpoint`` is set, because observability that leaves the
    machine is a decision the operator makes.

    ``capture_content`` writes the question, the answer, and the evidence into
    the local file instead of a fingerprint of them. It is a debugging switch:
    it is off unless asked for, and it stops on its own after
    ``capture_window_seconds`` so a debugging session cannot become the way the
    application runs. API keys are stripped either way.
    """

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    enabled: bool = Field(default=True, validation_alias="TELEMETRY_ENABLED")
    local_export_path: Path = Field(
        default=BACKEND_DIR / "data" / "traces" / "answer-traces.jsonl",
        validation_alias="TELEMETRY_LOCAL_EXPORT_PATH",
    )
    otlp_endpoint: str = Field(default="", validation_alias="TELEMETRY_OTLP_ENDPOINT")
    capture_content: bool = Field(
        default=False, validation_alias="TELEMETRY_CAPTURE_CONTENT"
    )
    capture_window_seconds: float = Field(
        default=900.0, validation_alias="TELEMETRY_CAPTURE_WINDOW_SECONDS"
    )

    @field_validator("enabled", "capture_content", mode="before")
    @classmethod
    def _coerce(cls, value):
        return _parse_bool(value)


class FrontendSettings(BaseSettings):
    """Frontend origin for CORS allowlist."""

    model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

    frontend_origin: str = Field(
        default="http://localhost:5173",
        validation_alias=AliasChoices(
            "FRONTEND_ORIGIN", "FRONTEND_URL", "CORS_ALLOWED_ORIGINS"
        ),
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
    query_context: QueryContextSettings = Field(
        default_factory=QueryContextSettings,
        validation_alias=AliasChoices("query_context_group"),
    )
    auth: AuthSettings = Field(default_factory=AuthSettings)
    upload: UploadSettings = Field(default_factory=UploadSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    research: ResearchSettings = Field(default_factory=ResearchSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
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

    if (
        not url_is_loopback(settings.vector.qdrant_url)
        and not (settings.vector.qdrant_api_key or "").strip()
    ):
        print(
            "PaperMind backend refuses a remote Qdrant address without an API key:",
            file=sys.stderr,
        )
        print(
            f"  - QDRANT_URL={settings.vector.qdrant_url!r} is not loopback, "
            "but QDRANT_API_KEY is unset.",
            file=sys.stderr,
        )
        print(
            "Fix: keep QDRANT_URL on http://localhost:6333 for local use, or set "
            "QDRANT_API_KEY to the same value as the Qdrant server's API key "
            "before exposing it beyond loopback (see backend/README.md).",
            file=sys.stderr,
        )
        sys.exit(1)

    warning = app_secret_warning(settings)
    if warning:
        print(warning, file=sys.stderr)
