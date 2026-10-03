"""Integration tests for settings, credentials, and research routes.

Needs ``pytest tests/integration``. Settings storage is faked but key
encryption is real, against the test secret.
"""

import pytest

from conftest import (
    FakeAppSettings,
    FakeFiles,
    make_client,
    make_ready_record,
    make_services,
)
from routes.credentials import resolve_stored_provider

pytestmark = pytest.mark.integration


def _record_pair(isettings):
    return [
        make_ready_record(isettings),
        make_ready_record(
            isettings,
            id="file-2",
            filename="doc2.pdf",
            title="Second Paper",
            original_filename="second.pdf",
        ),
    ]


class TestGetSettings:
    def test_empty(self, isettings):
        client = make_client(make_services(isettings), isettings)
        payload = client.get("/settings").get_json()
        assert payload["provider"] is None
        assert payload["model"] is None
        assert payload["masked_key"] is None
        assert "supported_models" in payload

    def test_stored(self, isettings, stored_settings):
        services = make_services(
            isettings, app_settings=FakeAppSettings(stored_settings)
        )
        payload = make_client(services, isettings).get("/settings").get_json()
        assert payload["provider"] == "google"
        assert payload["model"] == "gemini-2.5-flash"
        assert payload["masked_key"].endswith("••••") or "•" in payload["masked_key"]

    def test_corrupt_row(self, isettings):
        stored = {
            "provider": "google",
            "model": "retired-model",
            "encrypted_api_key": "x",
        }
        services = make_services(isettings, app_settings=FakeAppSettings(stored))
        response = make_client(services, isettings).get("/settings")
        assert response.status_code == 400
        payload = response.get_json()
        assert payload["category"] == "incomplete_settings"
        assert "supported_models" in payload["details"]
        assert payload["details"]["needs_resave"] is True


class TestPutSettings:
    def _body(self):
        return {"provider": "google", "model": "gemini-2.5-flash", "api_key": "key-1"}

    def test_invalid_body(self, isettings):
        services = make_services(isettings)
        response = make_client(services, isettings).put(
            "/settings", json={"provider": "google"}
        )
        assert response.status_code == 400
        assert response.get_json()["category"] == "invalid_settings"

    def test_unknown_model(self, isettings):
        services = make_services(isettings)
        response = make_client(services, isettings).put(
            "/settings", json={"provider": "google", "model": "nope", "api_key": "k"}
        )
        assert response.status_code == 400

    def test_verification_failure_keeps_old(self, isettings):
        services = make_services(
            isettings, verifier=lambda credentials: (False, "bad key")
        )
        response = make_client(services, isettings).put("/settings", json=self._body())
        assert response.status_code == 400
        assert response.get_json()["category"] == "settings_verification_failed"
        assert services.repositories.app_settings.upserts == []

    def test_verifier_crash_is_failure(self, isettings):
        def boom(credentials):
            raise RuntimeError("network down")

        services = make_services(isettings, verifier=boom)
        response = make_client(services, isettings).put("/settings", json=self._body())
        assert response.status_code == 400

    def test_success_stores_encrypted(self, isettings):
        from services.accounts.secrets_service import decrypt_api_key

        services = make_services(isettings)
        response = make_client(services, isettings).put("/settings", json=self._body())
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["provider"] == "google"
        (provider, model) = services.repositories.app_settings.upserts[0]
        assert (provider, model) == ("google", "gemini-2.5-flash")
        stored = services.repositories.app_settings.get_app_settings()
        assert decrypt_api_key(stored["encrypted_api_key"]) == "key-1"


class TestVerifySettings:
    def _body(self):
        return {"provider": "groq", "model": "openai/gpt-oss-20b", "api_key": "key-1"}

    def test_ok(self, isettings):
        services = make_services(isettings)
        response = make_client(services, isettings).post(
            "/settings/verify", json=self._body()
        )
        assert response.status_code == 200
        assert response.get_json()["ok"] is True

    def test_refused_key(self, isettings):
        services = make_services(isettings, verifier=lambda credentials: (False, "bad"))
        response = make_client(services, isettings).post(
            "/settings/verify", json=self._body()
        )
        assert response.status_code == 400
        payload = response.get_json()
        assert payload["category"] == "settings_verification_failed"
        assert payload["details"]["ok"] is False


class TestResolveStoredProvider:
    def test_no_row(self):
        with pytest.raises(Exception) as exc_info:
            resolve_stored_provider(FakeAppSettings(None), lambda creds: None)
        assert exc_info.value.category == "no_provider_configured"

    def test_bad_shape(self):
        with pytest.raises(Exception) as exc_info:
            resolve_stored_provider(
                FakeAppSettings({"provider": "google"}), lambda c: None
            )
        assert exc_info.value.category == "incomplete_settings"

    def test_unsupported_provider(self, stored_settings):
        stored = dict(stored_settings, provider="unknown")
        with pytest.raises(Exception) as exc_info:
            resolve_stored_provider(FakeAppSettings(stored), lambda c: None)
        assert exc_info.value.category == "unsupported_provider"

    def test_unsupported_model(self, stored_settings):
        stored = dict(stored_settings, model="retired")
        with pytest.raises(Exception) as exc_info:
            resolve_stored_provider(FakeAppSettings(stored), lambda c: None)
        assert exc_info.value.category == "unsupported_model"

    def test_factory_refusal(self, stored_settings):
        def factory(credentials):
            raise ValueError("nope")

        with pytest.raises(Exception) as exc_info:
            resolve_stored_provider(FakeAppSettings(stored_settings), factory)
        assert exc_info.value.category == "unsupported_provider"

    def test_bound(self, stored_settings, provider_model):
        binding = resolve_stored_provider(
            FakeAppSettings(stored_settings), lambda credentials: "provider-object"
        )
        assert binding.provider == "provider-object"
        assert binding.model == provider_model


class TestResearchValidation:
    def _client(self, isettings, stored):
        services = make_services(
            isettings,
            files=FakeFiles(_record_pair(isettings)),
            app_settings=FakeAppSettings(stored),
        )
        return make_client(services, isettings)

    def test_missing_question(self, isettings, stored_settings):
        response = self._client(isettings, stored_settings).post(
            "/research", json={"documents": ["doc1.pdf", "doc2.pdf"]}
        )
        assert response.status_code == 400
        assert response.get_json()["category"] == "invalid_question"

    def test_documents_not_list(self, isettings, stored_settings):
        response = self._client(isettings, stored_settings).post(
            "/research", json={"question": "q?", "documents": "doc1.pdf"}
        )
        assert response.status_code == 400

    def test_traversal_document(self, isettings, stored_settings):
        response = self._client(isettings, stored_settings).post(
            "/research", json={"question": "q?", "documents": ["../x.pdf", "doc2.pdf"]}
        )
        assert response.status_code == 400

    def test_single_document_refused(self, isettings, stored_settings):
        response = self._client(isettings, stored_settings).post(
            "/research", json={"question": "q?", "documents": ["doc1.pdf"]}
        )
        assert response.status_code == 400
        assert response.get_json()["category"] == "brief_scope_invalid"

    def test_unknown_document(self, isettings, stored_settings):
        response = self._client(isettings, stored_settings).post(
            "/research",
            json={"question": "q?", "documents": ["doc1.pdf", "missing.pdf"]},
        )
        assert response.status_code == 404

    def test_scope_ok(self, isettings, stored_settings):
        response = self._client(isettings, stored_settings).get(
            "/research/scope?documents=doc1.pdf&documents=doc2.pdf"
        )
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["document_count"] == 2
        assert [d["label"] for d in payload["documents"]] == ["A", "B"]

    def test_scope_refused(self, isettings, stored_settings):
        response = self._client(isettings, stored_settings).get(
            "/research/scope?documents=doc1.pdf"
        )
        assert response.status_code == 400

    def test_cancel_missing_id(self, isettings, stored_settings):
        response = self._client(isettings, stored_settings).post(
            "/research/cancel", json={}
        )
        assert response.status_code == 400

    def test_cancel_not_running(self, isettings, stored_settings):
        response = self._client(isettings, stored_settings).post(
            "/research/cancel", json={"brief_id": "never"}
        )
        assert response.status_code == 409
        assert response.get_json()["category"] == "brief_not_running"

    def test_cancel_running(self, isettings, stored_settings):
        from services.brief import cancellation

        cancellation.register("brief-1")
        try:
            response = self._client(isettings, stored_settings).post(
                "/research/cancel", json={"brief_id": "brief-1"}
            )
            assert response.status_code == 200
            assert response.get_json() == {"brief_id": "brief-1", "cancelled": True}
        finally:
            cancellation.release("brief-1")
