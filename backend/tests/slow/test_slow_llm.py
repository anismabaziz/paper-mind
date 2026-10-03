"""Slow tests for LLM provider construction and key verification.

Building a provider makes no network calls. Verifying a bogus key hits the
real API and must fail closed with (False, message). Needs
``pytest tests/slow`` with network access.
"""

import pytest

pytestmark = pytest.mark.slow


def _credentials(provider="groq", model="openai/gpt-oss-20b", api_key="bogus-key"):
    from services.accounts.chat_settings_service import model_for
    from services.llm.base import ChatCredentials

    definition = model_for("groq", "openai/gpt-oss-20b")
    return ChatCredentials(
        provider=provider,
        model=model,
        api_key=api_key,
        verification_timeout_seconds=5.0,
        budget=definition.chat_budget(),
    )


class TestProviderFactory:
    def test_unknown_provider_rejected_without_network(self):
        from services.llm.factory import build_chat_provider

        with pytest.raises(ValueError, match="Unsupported provider"):
            build_chat_provider(_credentials(provider="unknown"))

    def test_known_providers_build(self):
        from services.llm.factory import build_chat_provider

        assert build_chat_provider(_credentials(provider="groq")) is not None
        assert (
            build_chat_provider(
                _credentials(provider="google", model="gemini-2.5-flash")
            )
            is not None
        )


class TestKeyVerification:
    def test_bogus_key_fails_closed(self):
        from services.accounts.chat_settings_service import verify_api_key

        ok, message = verify_api_key(_credentials())
        assert ok is False
        assert message
