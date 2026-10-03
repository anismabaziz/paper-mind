"""Direct tests for AnswerService resolve and stream over fakes.

Needs ``pytest tests/integration``. Every terminal Turn outcome is covered:
answered, abstained, failed, cancelled, and persistence errors.
"""

import pytest

from conftest import (
    FakeConversations,
    FakeFiles,
    FakeIngestionJobs,
    FakeProvider,
    FakeVectorService,
    make_file_record,
    make_ready_record,
    make_services,
)
from services.answering import AnswerRequest, AnswerService
from services.retrieval.base import (
    RetrievalResult,
    VectorDimensionError,
    VectorStoreConfigurationError,
    VectorStoreUnavailableError,
)
from services.streaming import Refusal

pytestmark = pytest.mark.integration


def _source(content="The paper shows X."):
    return {
        "content": content,
        "document": "doc1.pdf",
        "chunk_index": 0,
        "page": 2,
        "content_hash": "h",
        "score": 0.9,
    }


def _claimed_answer(source_id="S1"):
    return (
        "The paper shows X.\n<claims>\n"
        f'{{"claim": "X holds", "sources": ["{source_id}"]}}\n'
        "</claims>"
    )


@pytest.fixture()
def answer_setup(isettings, provider_model):
    """A ready document with a conversation and a bound provider/model."""
    record = make_ready_record(isettings)
    state = {}

    class FlappingFiles(FakeFiles):
        def __init__(self):
            super().__init__([record])
            self.mode = "ready"

        def get_file(self, filename):
            if self.mode == "gone":
                return None
            current = super().get_file(filename)
            if self.mode == "deleting" and current:
                current["deletion_state"] = "deleting"
            return current

    files = FlappingFiles()
    conversations = FakeConversations()
    vectors = FakeVectorService(
        RetrievalResult(sources=[_source()], method="dense", outcome="success")
    )
    provider = FakeProvider(tokens=[_claimed_answer()])
    services = make_services(
        isettings,
        files=files,
        conversations=conversations,
        vectors=vectors,
        provider=provider,
    )
    service = AnswerService(
        settings=isettings,
        repositories=services.repositories,
        vector_service=vectors,
        tracer=services.tracer,
    )
    request = AnswerRequest(
        filename="doc1.pdf",
        query="What does it show?",
        provider=provider,
        model=provider_model,
    )
    state.update(
        files=files,
        conversations=conversations,
        vectors=vectors,
        provider=provider,
        service=service,
        settings=isettings,
    )
    return state, request


def _events(service, resolved):
    return list(service.stream(resolved))


def _names(events):
    return [event.name for event in events]


class TestResolve:
    def test_happy_path(self, answer_setup):
        state, request = answer_setup
        resolved = state["service"].resolve(request)
        assert not isinstance(resolved, Refusal)
        assert resolved.turn_id == "turn-1"
        assert resolved.sources[0]["source_id"] == "S1"
        assert resolved.abstention is None

    def test_unknown_document(self, answer_setup):
        state, request = answer_setup
        request = AnswerRequest(
            filename="missing.pdf",
            query=request.query,
            provider=request.provider,
            model=request.model,
        )
        refused = state["service"].resolve(request)
        assert refused.category == "file_not_found"

    def test_empty_query(self, answer_setup):
        state, request = answer_setup
        request = AnswerRequest(
            filename=request.filename,
            query="   ",
            provider=request.provider,
            model=request.model,
        )
        assert state["service"].resolve(request).category == "invalid_query"

    def test_too_long_query(self, answer_setup):
        state, request = answer_setup
        request = AnswerRequest(
            filename=request.filename,
            query="x" * 9000,
            provider=request.provider,
            model=request.model,
        )
        assert state["service"].resolve(request).category == "query_too_long"

    def test_no_conversation(self, answer_setup):
        state, request = answer_setup
        state["conversations"]._conversation_id = None
        assert state["service"].resolve(request).category == "conversation_not_found"

    def test_unprocessed_refused(self, isettings, provider_model, answer_setup):
        state, request = answer_setup
        record = make_file_record()
        service = AnswerService(
            settings=isettings,
            repositories=make_services(
                isettings, files=FakeFiles([record])
            ).repositories,
            vector_service=FakeVectorService(),
            tracer=state["service"]._tracer,
        )
        refused = service.resolve(
            AnswerRequest(
                filename="doc1.pdf",
                query="q?",
                provider=request.provider,
                model=request.model,
            )
        )
        assert refused.category == "index_pending"

    def test_store_unavailable(self, answer_setup):
        state, request = answer_setup
        state["vectors"]._result = None

        def boom(*args, **kwargs):
            raise VectorStoreUnavailableError("down")

        state["vectors"].retrieve = boom
        assert state["service"].resolve(request).category == "vector_store_unavailable"

    def test_store_misconfigured(self, answer_setup):
        state, request = answer_setup

        def boom(*args, **kwargs):
            raise VectorStoreConfigurationError("bad")

        state["vectors"].retrieve = boom
        assert (
            state["service"].resolve(request).category == "vector_store_configuration"
        )

    def test_dimension_mismatch(self, answer_setup):
        state, request = answer_setup

        def boom(*args, **kwargs):
            raise VectorDimensionError(expected=1024, got=512)

        state["vectors"].retrieve = boom
        refused = state["service"].resolve(request)
        assert refused.category == "vector_dimension_mismatch"
        assert refused.status == 409

    def test_deletion_race_refused(self, answer_setup):
        state, request = answer_setup
        state["files"].mode = "deleting"
        refused = state["service"].resolve(request)
        assert refused.category == "document_deleting"

    def test_start_turn_failure(self, answer_setup):
        state, request = answer_setup

        def boom(conversation_id, query):
            raise RuntimeError("db down")

        state["conversations"].start_turn = boom
        refused = state["service"].resolve(request)
        assert refused.status == 500


class TestStream:
    def test_answered(self, answer_setup):
        state, request = answer_setup
        events = _events(state["service"], state["service"].resolve(request))
        assert _names(events)[0] == "start"
        assert _names(events)[-1] == "done"
        done = events[-1].payload
        assert done["answer"] == "The paper shows X."
        assert done["grounded"] is True
        assert done["sources"][0]["source_id"] == "S1"
        assert ("complete", "turn-1", "The paper shows X.") in state[
            "conversations"
        ].calls

    def test_visible_tokens_hide_claims_block(self, answer_setup):
        state, request = answer_setup
        events = _events(state["service"], state["service"].resolve(request))
        tokens = "".join(e.payload["text"] for e in events if e.name == "token")
        assert "The paper shows X." in tokens
        assert "<claims>" not in tokens

    def test_abstain(self, answer_setup):
        state, request = answer_setup
        state["vectors"]._result = RetrievalResult(
            sources=[], method="dense", outcome="empty"
        )
        events = _events(state["service"], state["service"].resolve(request))
        assert _names(events) == ["start", "abstained"]
        assert events[1].payload["abstained"] is True

    def test_abstain_persist_failure(self, answer_setup):
        state, request = answer_setup
        state["vectors"]._result = RetrievalResult(
            sources=[], method="dense", outcome="empty"
        )
        state["conversations"].abstain_turn = lambda *a: False
        events = _events(state["service"], state["service"].resolve(request))
        assert _names(events) == ["persistence_error"]

    def test_provider_timeout(self, answer_setup):
        from services.llm.base import ProviderTimeoutError

        state, request = answer_setup

        def slow(query, context, prior_turns=""):
            raise ProviderTimeoutError("slow")
            yield

        state["provider"].stream_response = slow
        events = _events(state["service"], state["service"].resolve(request))
        assert _names(events)[-1] == "provider_error"
        assert ("fail", "turn-1", "provider timeout") in state["conversations"].calls

    def test_empty_answer(self, answer_setup):
        from services.llm.base import EmptyAnswerError

        state, request = answer_setup

        def silent(query, context, prior_turns=""):
            raise EmptyAnswerError()
            yield

        state["provider"].stream_response = silent
        events = _events(state["service"], state["service"].resolve(request))
        assert _names(events)[-1] == "provider_error"

    def test_provider_crash(self, answer_setup):
        state, request = answer_setup

        def broken(query, context, prior_turns=""):
            raise RuntimeError("api exploded")
            yield

        state["provider"].stream_response = broken
        events = _events(state["service"], state["service"].resolve(request))
        assert _names(events)[-1] == "provider_error"
        assert ("fail", "turn-1", "provider failure") in state["conversations"].calls

    def test_citation_repair_then_done(self, answer_setup):
        state, request = answer_setup
        state["provider"]._tokens = [_claimed_answer("S9")]
        state["provider"]._repair = _claimed_answer("S1")
        events = _events(state["service"], state["service"].resolve(request))
        assert _names(events)[-1] == "done"

    def test_unresolved_citations_fail_turn(self, answer_setup):
        state, request = answer_setup
        state["provider"]._tokens = [_claimed_answer("S9")]
        state["provider"]._repair = _claimed_answer("S7")
        events = _events(state["service"], state["service"].resolve(request))
        assert _names(events)[-1] == "citation_error"
        assert ("fail", "turn-1", "invalid citations") in state["conversations"].calls

    def test_complete_turn_refused_is_persistence_error(self, answer_setup):
        state, request = answer_setup
        state["conversations"].complete_turn = lambda *a, **k: False
        events = _events(state["service"], state["service"].resolve(request))
        assert _names(events)[-1] == "persistence_error"

    def test_complete_turn_crash_is_persistence_error(self, answer_setup):
        state, request = answer_setup

        def boom(*args, **kwargs):
            raise RuntimeError("db down")

        state["conversations"].complete_turn = boom
        events = _events(state["service"], state["service"].resolve(request))
        assert _names(events)[-1] == "persistence_error"

    def test_client_disconnect_cancels(self, answer_setup):
        state, request = answer_setup
        resolved = state["service"].resolve(request)
        stream = state["service"].stream(resolved)
        assert next(stream).name == "start"
        assert next(stream).name == "token"
        stream.close()
        assert ("cancel", "turn-1", "client disconnected") in state[
            "conversations"
        ].calls

    def test_document_deleted_mid_answer_cancels(self, answer_setup):
        state, request = answer_setup
        resolved = state["service"].resolve(request)
        state["files"].mode = "gone"
        events = _events(state["service"], resolved)
        assert _names(events)[-1] == "cancelled"
