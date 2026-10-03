"""Integration tests for chat routes over fakes.

Needs ``pytest tests/integration``. Covers refusal mapping, the SSE happy
path, abstention, and message history.
"""

import pytest

from conftest import (
    FakeAppSettings,
    FakeConversations,
    FakeFiles,
    FakeProvider,
    FakeVectorService,
    make_client,
    make_pdf_bytes,
    make_ready_record,
    make_services,
)
from services.retrieval.base import RetrievalResult

pytestmark = pytest.mark.integration


def _source(content="Relevant passage text here."):
    return {
        "content": content,
        "document": "doc1.pdf",
        "chunk_index": 0,
        "page": 2,
        "content_hash": "h",
        "score": 0.9,
    }


def _answer_tokens():
    return [
        "The paper shows X.\n<claims>\n",
        '{"claim": "X holds", "sources": ["S1"]}\n',
        "</claims>",
    ]


def _chat_services(isettings, stored, **overrides):
    record = make_ready_record(isettings)
    services = make_services(
        isettings,
        files=FakeFiles([record]),
        conversations=FakeConversations(),
        app_settings=FakeAppSettings(stored),
        **overrides,
    )
    services.storage.save(record["filename"], make_pdf_bytes())
    return services


class TestResponseValidation:
    def test_missing_fields(self, isettings, stored_settings):
        client = make_client(_chat_services(isettings, stored_settings), isettings)
        response = client.post("/response", json={"query": "q?"})
        assert response.status_code == 400
        assert response.get_json()["category"] == "invalid_query"

    def test_traversal_filename(self, isettings, stored_settings):
        client = make_client(_chat_services(isettings, stored_settings), isettings)
        response = client.post(
            "/response", json={"query": "q?", "filename": "../x.pdf"}
        )
        assert response.status_code == 400
        assert response.get_json()["category"] == "invalid_filename"

    def test_no_provider_configured(self, isettings):
        client = make_client(_chat_services(isettings, None), isettings)
        response = client.post(
            "/response", json={"query": "q?", "filename": "doc1.pdf"}
        )
        assert response.status_code == 400
        assert response.get_json()["category"] == "no_provider_configured"

    def test_unknown_document(self, isettings, stored_settings):
        client = make_client(_chat_services(isettings, stored_settings), isettings)
        response = client.post(
            "/response", json={"query": "q?", "filename": "missing.pdf"}
        )
        assert response.status_code == 404

    def test_pending_document(self, isettings, stored_settings):
        from conftest import make_file_record

        services = make_services(
            isettings,
            files=FakeFiles([make_file_record()]),
            app_settings=FakeAppSettings(stored_settings),
        )
        response = make_client(services, isettings).post(
            "/response", json={"query": "q?", "filename": "doc1.pdf"}
        )
        assert response.status_code == 409
        assert response.get_json()["category"] == "index_pending"

    def test_stale_document(self, isettings, stored_settings):
        from conftest import make_file_record

        stale = make_file_record(is_processed=True, index_manifest=None)
        services = make_services(
            isettings,
            files=FakeFiles([stale]),
            app_settings=FakeAppSettings(stored_settings),
        )
        response = make_client(services, isettings).post(
            "/response", json={"query": "q?", "filename": "doc1.pdf"}
        )
        assert response.status_code == 409
        assert response.get_json()["category"] == "index_stale"

    def test_deleting_document(self, isettings, stored_settings):
        record = make_ready_record(isettings, deletion_state="deleting")
        services = make_services(
            isettings,
            files=FakeFiles([record]),
            app_settings=FakeAppSettings(stored_settings),
        )
        response = make_client(services, isettings).post(
            "/response", json={"query": "q?", "filename": "doc1.pdf"}
        )
        assert response.status_code == 409


class TestResponseStream:
    def test_happy_path_done(self, isettings, stored_settings):
        vectors = FakeVectorService(
            RetrievalResult(sources=[_source()], method="dense", outcome="success")
        )
        services = _chat_services(
            isettings,
            stored_settings,
            vectors=vectors,
            provider=FakeProvider(tokens=_answer_tokens()),
        )
        response = make_client(services, isettings).post(
            "/response", json={"query": "What does it show?", "filename": "doc1.pdf"}
        )
        assert response.status_code == 200
        body = response.get_data(as_text=True)
        assert "event: start" in body
        assert "event: done" in body
        assert '"grounded": true' in body
        kinds = [c[0] for c in services.repositories.conversations.calls]
        assert "start" in kinds and "complete" in kinds

    def test_abstention_when_no_evidence(self, isettings, stored_settings):
        services = _chat_services(isettings, stored_settings)
        response = make_client(services, isettings).post(
            "/response", json={"query": "What does it show?", "filename": "doc1.pdf"}
        )
        body = response.get_data(as_text=True)
        assert "event: abstained" in body
        assert "event: done" not in body

    def test_provider_timeout_is_provider_error(self, isettings, stored_settings):
        from services.llm.base import ProviderTimeoutError

        class SlowProvider(FakeProvider):
            def stream_response(self, query, context, prior_turns=""):
                raise ProviderTimeoutError("slow")
                yield

        vectors = FakeVectorService(
            RetrievalResult(sources=[_source()], method="dense", outcome="success")
        )
        services = _chat_services(
            isettings, stored_settings, vectors=vectors, provider=SlowProvider()
        )
        body = (
            make_client(services, isettings)
            .post("/response", json={"query": "q?", "filename": "doc1.pdf"})
            .get_data(as_text=True)
        )
        assert "event: provider_error" in body


class TestMessages:
    def test_missing_filename(self, isettings):
        client = make_client(make_services(isettings), isettings)
        assert client.get("/messages").status_code == 400

    def test_unknown_file_empty(self, isettings):
        client = make_client(make_services(isettings), isettings)
        response = client.get("/messages?filename=no.pdf")
        assert response.status_code == 200
        assert response.get_json() == {"messages": []}

    def test_no_conversation_empty(self, isettings):
        services = make_services(
            isettings,
            files=FakeFiles([make_ready_record(isettings)]),
            conversations=FakeConversations(conversation_id=None),
        )
        response = make_client(services, isettings).get("/messages?filename=doc1.pdf")
        assert response.get_json() == {"messages": []}

    def test_returns_messages(self, isettings):
        messages = [{"id": "m1", "text": "hi", "sender": "user"}]
        services = make_services(
            isettings,
            files=FakeFiles([make_ready_record(isettings)]),
            conversations=FakeConversations(messages=messages),
        )
        response = make_client(services, isettings).get("/messages?filename=doc1.pdf")
        assert response.get_json() == {"messages": messages}
