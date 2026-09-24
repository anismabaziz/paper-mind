"""Production dependency graph for the Flask application."""

from collections.abc import Callable
from dataclasses import dataclass

from providers import get_vector_index
from repositories import Repositories, build_repositories
from services.accounts.chat_settings_service import verify_api_key
from services.embeddings.local_embeddings import EmbeddingService, LocalEmbeddingService
from services.llm.base import ChatCredentials, LLMProvider
from services.llm.factory import build_chat_provider
from services.parsing.document_parser import DocumentIngestor
from services.retrieval.reranker import RerankerService
from services.retrieval.vector_service import VectorService
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

    @classmethod
    def from_settings(cls, app_settings: Settings) -> "Services":
        """Build the production graph from validated settings."""
        return cls(
            settings=app_settings,
            repositories=build_repositories(),
            storage=get_storage(),
            parser=DocumentIngestor(
                app_settings.parsing.use_docling,
                app_settings.chunking.chunk_size_tokens,
                app_settings.chunking.chunk_overlap_tokens,
            ),
            embedding_service=LocalEmbeddingService(
                app_settings.embedding.embedding_model
            ),
            vector_service=VectorService(
                get_vector_index(),
                RerankerService(
                    app_settings.rerank.rerank_model,
                    enabled=app_settings.rerank.enabled,
                ),
            ),
            chat_provider_factory=build_chat_provider,
            api_key_verifier=verify_api_key,
        )
