"""
The evaluation environment: the application, with the fixture documents in it.

A run is only worth reading if it describes the app, so a run is set up the
way the app is: the fixture documents are stored and indexed through the same
ingestion job and worker that the upload route uses, which gives each one a
real Conversation, a real Index Generation, and a real Index Manifest, and the
questions are then asked through the same answer path the chat route uses.

What a run chooses for itself is only what a test has to be able to choose:
the store the documents live in, the embedding and provider that serve them,
and the clock that measures them. None of that is a branch in the answer path.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from repositories import Repositories, build_repositories
from services.accounts.chat_settings_service import ModelCapabilities
from services.answering import AnswerService
from services.embeddings.local_embeddings import EmbeddingService
from services.ingestion.worker import IngestionWorker
from services.llm.base import ChatCredentials, LLMProvider
from services.parsing.document_parser import DocumentIngestor
from services.retrieval.base import RetrievalResult, VectorStoreConfigurationError
from services.retrieval.vector_service import VectorService
from settings import Settings

SAMPLE_DOCS_DIR = Path(__file__).parent / "sample_docs"

#: The question a retrieval capability check asks. It is ordinary text with
#: content words in it, so the production path selects hybrid retrieval for it
#: and a store that cannot serve hybrid cannot pass the check.
CONTRACT_PROBE_QUERY = "retrieval contract probe"


def read_document(filename: str, docs_dir: Path = SAMPLE_DOCS_DIR) -> bytes:
    """Return the bytes of one fixture document."""
    return (Path(docs_dir) / filename).read_bytes()


@dataclass
class EvaluationEnvironment:
    """
    Everything a run asks questions through.

    The environment holds the production collaborators and the mapping from a
    fixture document to the stored Document the application knows about. A
    live run prefixes the stored names so a measured run never answers from —
    or deletes — vectors belonging to a reader's own library.
    """

    settings: Settings
    repositories: Repositories
    storage: Any
    parser: DocumentIngestor
    embedding_service: EmbeddingService
    vector_service: VectorService
    answer_service: AnswerService
    chat_provider_factory: Callable[[ChatCredentials], LLMProvider]
    documents: dict[str, str] = field(default_factory=dict)
    #: Wall time each Document took to become answerable, keyed by fixture name.
    indexed_seconds: dict[str, float] = field(default_factory=dict)
    worker: IngestionWorker | None = None
    #: The clock the run measures with, held here because a report's latency is
    #: a measurement of something and the measurement is an input, not a
    #: decision the answer path makes.
    clock: Callable[[], float] = time.monotonic

    def stored_name(self, fixture_filename: str) -> str:
        """Return the stored Document a fixture document was indexed as."""
        try:
            return self.documents[fixture_filename]
        except KeyError:
            raise KeyError(f"{fixture_filename} was not indexed for this run") from None

    def provider(
        self, model: ModelCapabilities, api_key: str = "evaluation"
    ) -> LLMProvider:
        """Return the provider this run answers with, through the app's factory."""
        return self.chat_provider_factory(
            ChatCredentials(
                provider=model.provider,
                model=model.id,
                api_key=api_key,
                verification_timeout_seconds=model.timeout_seconds,
                budget=model.chat_budget(),
            )
        )

    def conversation_id(self, fixture_filename: str) -> str | None:
        """Return the Conversation a fixture document's questions are recorded in."""
        record = self.repositories.files.get_file(self.stored_name(fixture_filename))
        if record is None:
            return None
        return self.repositories.conversations.get_conversation_id(record["id"])

    def seed_turns(self, fixture_filename: str, questions: Sequence[str]) -> None:
        """
        Record earlier questions on a Document's Conversation.

        A multi-turn case refers back to exchanges the reader already had, so
        the run commits them as answered Turns and lets the answer path read
        them back the way it reads any other recorded Turns. Nothing is
        injected into the prompt directly.
        """
        conversation_id = self.conversation_id(fixture_filename)
        if conversation_id is None or not questions:
            return
        for question in questions:
            turn_id = self.repositories.conversations.start_turn(
                conversation_id, question
            )
            self.repositories.conversations.complete_turn(
                turn_id, "Answer recorded before this question.", []
            )

    def index_state(self, fixture_filename: str) -> dict[str, Any]:
        """
        Return what the run knows about a Document's index.

        The manifest and generation say what the vectors were built with, and
        the seconds say how long indexing it took, so a run reports the cost of
        preparing its evidence as well as the quality of it.
        """
        record = self.repositories.files.get_file(self.stored_name(fixture_filename))
        return {
            "index_manifest": (record or {}).get("index_manifest"),
            "index_generation": (record or {}).get("index_generation"),
            "is_processed": bool((record or {}).get("is_processed")),
            "indexed_seconds": self.indexed_seconds.get(fixture_filename),
        }


def require_hybrid_retrieval(
    environment: EvaluationEnvironment, filename: str
) -> RetrievalResult:
    """
    Prove the store serves the retrieval the app selects, or fail the run.

    A store that quietly answers a hybrid question with dense results would
    make every reported number describe a system nobody runs. The check asks
    through the production service and reads the method it reports, so a
    degraded double stops the run instead of quietly lowering the scores.
    """
    record = environment.repositories.files.get_file(filename)
    if record is None:
        raise VectorStoreConfigurationError(
            f"{filename} is not stored, so retrieval cannot be checked"
        )
    embedding = environment.embedding_service.embed_texts(CONTRACT_PROBE_QUERY)[0]
    result = environment.vector_service.query_vectors(
        embedding,
        filename,
        query_text=CONTRACT_PROBE_QUERY,
        generation=record.get("index_generation"),
        include_legacy=record.get("index_generation") is None,
    )
    if result.method != "hybrid":
        raise VectorStoreConfigurationError(
            f"the vector store answered a hybrid question with {result.method} "
            f"results for {filename}; this run would not describe the app"
        )
    return result


def build_environment(
    fixture: dict,
    *,
    settings: Settings,
    session_factory: Any,
    storage: Any,
    embedding_service: EmbeddingService,
    vector_service: VectorService,
    chat_provider_factory: Callable[[ChatCredentials], LLMProvider],
    documents_prefix: str = "",
    docs_dir: Path = SAMPLE_DOCS_DIR,
    clock: Callable[[], float] = time.monotonic,
    worker_id: str = "evaluation-worker",
) -> EvaluationEnvironment:
    """
    Index the fixture documents into a working application and return it.

    Each document is stored, given a first ingestion job, and processed by the
    production worker, so what the run retrieves is what a reader who uploaded
    the same Document would retrieve. The retrieval contract is checked once,
    here, rather than per question: a store that cannot serve it fails the run.
    """
    from services.indexing.manifest import manifest_builder

    repositories = build_repositories(session_factory)
    answer_service = AnswerService(
        settings=settings,
        repositories=repositories,
        embedding_service=embedding_service,
        vector_service=vector_service,
        clock=clock,
    )
    parser = DocumentIngestor(
        settings.parsing.use_docling,
        settings.chunking.chunk_size_tokens,
        settings.chunking.chunk_overlap_tokens,
    )
    worker = IngestionWorker(
        repositories=repositories,
        storage=storage,
        parser=parser,
        embedding_service=embedding_service,
        vector_service=vector_service,
        worker_id=worker_id,
        limits=settings.ingestion,
        manifest_builder=manifest_builder(settings),
        clock=clock,
    )
    documents: dict[str, str] = {}
    indexed_seconds: dict[str, float] = {}
    for document in fixture.get("documents", []):
        filename = document["filename"]
        stored = f"{documents_prefix}{filename}"
        storage.save(stored, read_document(filename, docs_dir))
        repositories.files.create_file_with_job(
            stored, title=filename, original_filename=filename
        )
        documents[filename] = stored
    # One drain covers every document the run queued, so the seconds are read
    # per document afterwards from when each job was claimed to ready.
    worker.drain()
    for filename in documents:
        indexed_seconds[filename] = _indexed_seconds(repositories, documents[filename])

    environment = EvaluationEnvironment(
        settings=settings,
        repositories=repositories,
        storage=storage,
        parser=parser,
        embedding_service=embedding_service,
        vector_service=vector_service,
        answer_service=answer_service,
        chat_provider_factory=chat_provider_factory,
        documents=documents,
        indexed_seconds=indexed_seconds,
        worker=worker,
        clock=clock,
    )
    for stored in documents.values():
        record = repositories.files.get_file(stored)
        if not record or not record.get("is_processed"):
            raise VectorStoreConfigurationError(
                f"{stored} was not indexed for this run, so no case can ask it"
            )
        require_hybrid_retrieval(environment, stored)
    return environment


def _indexed_seconds(repositories: Repositories, stored: str) -> float:
    """Return how long the last ingestion job for a Document took."""
    job = repositories.ingestion_jobs.get_latest(stored) or {}
    started = job.get("started_at")
    finished = job.get("finished_at")
    if not started or not finished:
        return 0.0
    try:
        return max(
            0.0,
            (
                datetime.fromisoformat(finished) - datetime.fromisoformat(started)
            ).total_seconds(),
        )
    except (TypeError, ValueError):
        return 0.0


def remove_documents(environment: EvaluationEnvironment) -> None:
    """Delete the run's Documents and their vectors, leaving the app as it was."""
    files = environment.repositories.files
    for stored in environment.documents.values():
        record = files.get_file(stored)
        if record is not None:
            files.delete_file(record["id"])
        try:
            environment.vector_service.delete_by_filename(stored)
        except Exception:  # noqa: BLE001 - cleanup is best effort per document
            pass
