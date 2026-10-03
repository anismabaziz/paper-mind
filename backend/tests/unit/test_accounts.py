"""Unit tests for provider settings and key encryption.

Encryption runs against test settings installed globally for the test only.
Run with plain ``pytest``.
"""

import pytest

import settings as settings_module
from services.accounts.chat_settings_service import (
    SUPPORTED_MODELS,
    SettingsError,
    mask_key,
    model_for,
    supported_models_payload,
    verification_error_message,
)
from services.accounts.model_availability import (
    format_report,
    keys_from_env,
    provider_api_key_env,
)
from services.accounts.secrets_service import decrypt_api_key, encrypt_api_key

pytestmark = pytest.mark.unit


@pytest.fixture()
def _test_secret():
    original = settings_module.get_settings() if _has_settings() else None
    candidate = settings_module.Settings()
    candidate.auth.app_secret = "test-secret-for-unit-tests"
    settings_module.set_settings(candidate)
    yield candidate
    settings_module.set_settings(original)


def _has_settings():
    try:
        settings_module.get_settings()
        return True
    except Exception:
        return False


class TestModelCatalog:
    def test_supported_providers(self):
        assert set(SUPPORTED_MODELS) == {"google", "groq"}

    def test_model_for_known(self):
        model = model_for("google", "gemini-2.5-flash")
        assert model.provider == "google"
        assert model.id == "gemini-2.5-flash"
        assert model.max_input_tokens > 0

    def test_model_for_unknown_model(self):
        with pytest.raises(SettingsError):
            model_for("google", "nope")

    def test_model_for_unknown_provider(self):
        with pytest.raises(SettingsError):
            model_for("nope", "nope")

    def test_mask_key(self):
        assert mask_key("sk-abcdef123456") == "••••3456"
        assert mask_key("") == "••••"

    def test_catalog_payload_shape(self):
        payload = supported_models_payload()
        assert set(payload) == {"google", "groq"}

    def test_verification_message_from_exception(self):
        assert "Check the values" in verification_error_message(ValueError("x"))

    def test_verification_message_classifies_key_failures(self):
        assert "rejected the API key" in verification_error_message(
            "invalid api key 401"
        )

    def test_verification_message_classifies_network_failures(self):
        assert "did not respond" in verification_error_message("connection timeout")

    def test_verification_message_classifies_model_failures(self):
        assert "supported model" in verification_error_message("model not found 404")

    def test_verification_message_generic_fallback(self):
        assert "Check the values" in verification_error_message("custom detail")


class TestSecretsService:
    def test_roundtrip(self, _test_secret):
        assert decrypt_api_key(encrypt_api_key("my-key")) == "my-key"

    def test_ciphertexts_differ(self, _test_secret):
        assert encrypt_api_key("same") != encrypt_api_key("same")

    def test_garbage_ciphertext_needs_resave_or_error(self, _test_secret):
        from services.accounts.secrets_service import SecretsError

        with pytest.raises(SecretsError):
            decrypt_api_key("not-a-ciphertext")


class TestModelAvailabilityPure:
    def test_keys_from_env(self):
        assert keys_from_env({}) == {}
        assert keys_from_env(None) == {}
        keys = keys_from_env(
            {
                "PAPERMIND_CHECK_GOOGLE_API_KEY": "g",
                "PAPERMIND_CHECK_GROQ_API_KEY": "q",
                "OTHER": "x",
            }
        )
        assert keys == {"google": "g", "groq": "q"}

    def test_provider_env_names(self):
        assert provider_api_key_env("google") == "PAPERMIND_CHECK_GOOGLE_API_KEY"
        assert provider_api_key_env("groq") == "PAPERMIND_CHECK_GROQ_API_KEY"
        assert provider_api_key_env("unknown") is None

    def test_format_report_ok(self):
        from services.accounts.model_availability import AvailabilityReport

        report = AvailabilityReport(checked=(("google", "m"),), checked_count=1)
        assert report.ok is True
        assert "1 catalog models are available" in format_report(report)

    def test_format_report_missing(self):
        from services.accounts.model_availability import AvailabilityReport

        report = AvailabilityReport(missing_keys=("PAPERMIND_CHECK_GOOGLE_API_KEY",))
        assert report.ok is False
        assert "PAPERMIND_CHECK_GOOGLE_API_KEY" in format_report(report)

    def test_check_catalog_with_fake_factory(self):
        from services.accounts.chat_settings_service import MODEL_CATALOG
        from services.accounts.model_availability import check_catalog

        class FakeProvider:
            def __init__(self, key):
                self._key = key

            def verify(self):
                if self._key == "bad":
                    raise RuntimeError("bad key")
                return True

        def factory(credentials, use_cache=True):
            return FakeProvider(credentials.api_key)

        report = check_catalog(
            {"google": "good", "groq": "bad"},
            provider_factory=factory,
            catalog=MODEL_CATALOG,
        )
        assert report.checked_count == len(MODEL_CATALOG)
        assert all(provider == "groq" for provider, _ in report.unavailable)
        assert report.ok is False

    def test_check_catalog_missing_key_skips_provider(self):
        from services.accounts.chat_settings_service import MODEL_CATALOG
        from services.accounts.model_availability import check_catalog

        report = check_catalog({}, catalog=MODEL_CATALOG)
        assert report.checked_count == 0
        assert report.missing_keys
        assert report.ok is False

    def test_check_catalog_factory_crash_is_finding(self):
        from services.accounts.chat_settings_service import MODEL_CATALOG
        from services.accounts.model_availability import check_catalog

        def factory(credentials, use_cache=True):
            raise RuntimeError("factory exploded")

        report = check_catalog(
            {"google": "k", "groq": "k"},
            provider_factory=factory,
            catalog=MODEL_CATALOG,
        )
        assert len(report.unavailable) == len(MODEL_CATALOG)
        assert len(report.reasons) == len(MODEL_CATALOG)
