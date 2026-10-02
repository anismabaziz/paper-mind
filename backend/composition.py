"""Production dependency graph for the Flask application."""

from collections.abc import Callable
from dataclasses import dataclass, field

from providers import get_vector_index
from repositories import Repositories, build_repositories
from services.accounts.chat_settings_service import verify_api_key
from services.embeddings.local_embeddings import EmbeddingService, LocalEmbeddingService
from services.llm.base import ChatCredentials, LLMProvider
from services.llm.factory import build_chat_provider
from services.parsing.document_parser import DocumentIngestor
from services.retrieval.reranker import RerankerService
from services.retrieval.vector_service import VectorService
from services.telemetry.factory import tracer_for
from services.telemetry.spans import Tracer
from settings import Settings
from storage import LocalStorage, get_storage

CredentialsCallable = Callable[[ChatCredentials], LLMProvider]
VerifyCallable = Callable[[ChatCredentials], tuple[bool, str | None]]


@dataclass(frozen=True)
class Services:
    """Dependencies used by the HTTP routes."""

    settings: Settings
    repositories: Repositories
    storage: LocalStorage
    parser: DocumentIngestor
    embedding_service: EmbeddingService
    vector_service: VectorService
    chat_provider_factory: CredentialsCallable
    api_key_verifier: VerifyCallable
    #: Where each answer request's trace goes. It is a dependency rather than
    #: something a route builds, so a deployment configures observability in
    #: settings and a test can read what the application would have exported.
    tracer: Tracer = field(default_factory=Tracer)

    @classmethod
    def from_settings(cls, app_settings: Settings) -> "Services":
        """Build the production graph from validated settings."""
        embedding_service = LocalEmbeddingService(
            app_settings.embedding.embedding_model,
            revision=app_settings.embedding.revision,
            trust_remote_code=app_settings.embedding.trust_remote_code,
        )
        return cls(
            settings=app_settings,
            repositories=build_repositories(),
            storage=get_storage(),
            parser=DocumentIngestor(
                app_settings.parsing.use_docling,
                app_settings.chunking.chunk_size_tokens,
                app_settings.chunking.chunk_overlap_tokens,
            ),
            embedding_service=embedding_service,
            vector_service=VectorService(
                get_vector_index(),
                RerankerService(
                    app_settings.rerank.rerank_model,
                    enabled=app_settings.rerank.enabled,
                    revision=app_settings.rerank.revision,
                    trust_remote_code=app_settings.rerank.trust_remote_code,
                ),
                embedding_service=embedding_service,
            ),
            chat_provider_factory=build_chat_provider,
            api_key_verifier=verify_api_key,
            tracer=tracer_for(app_settings),
        )
