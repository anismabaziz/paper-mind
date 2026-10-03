"""Unit tests for deletion guards and the error shape.

Covers services.deletion and errors without a Flask app, except the handler
registration which uses a bare app. Run with plain ``pytest``.
"""

import pytest

from errors import (
    AppError,
    BadRequest,
    Conflict,
    InternalError,
    NotFound,
    PayloadTooLarge,
    ServiceUnavailable,
    UnsupportedMediaType,
    category_for_status,
    classify,
    error_payload,
    payload_for,
    refusal,
    register,
    register_domain_errors,
)
from services.deletion import deletion_block_payload, is_deleting_record

pytestmark = pytest.mark.unit


class TestIsDeletingRecord:
    @pytest.mark.parametrize("state", ["deleting", "delete_failed"])
    def test_deleting_states(self, state):
        assert is_deleting_record({"deletion_state": state}) is True

    @pytest.mark.parametrize("state", ["active", None])
    def test_live_states(self, state):
        assert is_deleting_record({"deletion_state": state}) is False

    def test_none_record(self):
        assert is_deleting_record(None) is False


class TestDeletionBlockPayload:
    def test_deleting(self):
        payload = deletion_block_payload({"deletion_state": "deleting"})
        assert payload["category"] == "document_deleting"
        assert payload["deletion_state"] == "deleting"

    def test_delete_failed(self):
        payload = deletion_block_payload(
            {"deletion_state": "delete_failed", "deletion_error": "boom"}
        )
        assert payload["category"] == "document_delete_failed"
        assert payload["deletion_error"] == "boom"

    def test_none_defaults_to_deleting(self):
        assert deletion_block_payload(None)["category"] == "document_deleting"


class TestErrorClasses:
    @pytest.mark.parametrize(
        ("cls", "status", "category"),
        [
            (BadRequest, 400, "invalid_request"),
            (NotFound, 404, "not_found"),
            (Conflict, 409, "conflict"),
            (PayloadTooLarge, 413, "payload_too_large"),
            (UnsupportedMediaType, 415, "unsupported_media_type"),
            (InternalError, 500, "internal"),
            (ServiceUnavailable, 503, "service_unavailable"),
        ],
    )
    def test_status_and_category(self, cls, status, category):
        error = cls("message")
        assert error.status == status
        assert error.category == category
        assert error.details == {}

    def test_custom_category_and_details(self):
        error = BadRequest("bad", category="file_empty", details={"x": 1})
        assert error.category == "file_empty"
        assert error.details == {"x": 1}

    def test_default_message(self):
        assert "Internal" in AppError().message


class TestRefusalAndPayload:
    def test_refusal_shape(self):
        error = refusal("nope", "invalid_query", 400)
        assert error.status == 400
        assert error.category == "invalid_query"
        assert payload_for(error) == {
            "error": "nope",
            "category": "invalid_query",
            "details": {},
        }

    def test_refusal_falls_back_to_status_category(self):
        assert refusal("nope", "", 404).category == "not_found"

    def test_error_payload_defaults_details(self):
        assert error_payload("c", "m") == {"error": "m", "category": "c", "details": {}}

    @pytest.mark.parametrize(
        ("status", "category"),
        [
            (400, "invalid_request"),
            (404, "not_found"),
            (409, "conflict"),
            (503, "service_unavailable"),
        ],
    )
    def test_known_statuses(self, status, category):
        assert category_for_status(status) == category

    def test_unknown_status_splits_client_server(self):
        assert category_for_status(418) == "http_error"
        assert category_for_status(599) == "internal"


class TestClassify:
    def test_app_error_passes_through(self):
        error = Conflict("busy", category="x")
        assert classify(error) is error

    def test_unknown_becomes_internal(self):
        classified = classify(RuntimeError("boom"))
        assert isinstance(classified, InternalError)

    def test_registered_domain_error(self):
        class LocalFailure(Exception):
            pass

        register(LocalFailure, lambda e: BadRequest("local", category="local_fail"))
        classified = classify(LocalFailure())
        assert isinstance(classified, BadRequest)
        assert classified.category == "local_fail"

    def test_subclass_registration_wins_over_parent(self):
        class Parent(Exception):
            pass

        class Child(Parent):
            pass

        register(Parent, lambda e: BadRequest("parent"))
        register(Child, lambda e: Conflict("child"))
        assert isinstance(classify(Child()), Conflict)

    def test_domain_registry_known_mappings(self):
        from repositories.ingestion_jobs import JobConflictError
        from services.retrieval.base import (
            VectorDimensionError,
            VectorStoreConfigurationError,
            VectorStoreUnavailableError,
        )

        register_domain_errors()
        assert classify(JobConflictError()).category == "ingestion_job_active"
        assert classify(VectorStoreUnavailableError()).status == 503
        assert classify(VectorStoreConfigurationError()).status == 500
        assert classify(VectorDimensionError()).status == 409

    def test_flask_handler_shape(self):
        from flask import Flask

        from errors import register_error_handlers

        register_domain_errors()
        app = Flask(__name__)
        register_error_handlers(app)

        @app.route("/boom")
        def boom():
            raise Conflict("busy", category="doc_busy", details={"left": 1})

        client = app.test_client()
        response = client.get("/boom")
        assert response.status_code == 409
        assert response.get_json() == {
            "error": "busy",
            "category": "doc_busy",
            "details": {"left": 1},
        }

    def test_unhandled_exception_is_json_not_html(self):
        from flask import Flask

        from errors import register_error_handlers

        app = Flask(__name__)
        register_error_handlers(app)

        @app.route("/crash")
        def crash():
            raise RuntimeError("unexpected")

        response = app.test_client().get("/crash")
        assert response.status_code == 500
        assert response.get_json()["category"] == "internal"
