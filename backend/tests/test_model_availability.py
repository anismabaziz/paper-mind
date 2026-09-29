"""
Model catalog availability check.

A catalog entry is a promise that a user can pick that model and get an
answer. Providers retire models without warning, and a shipped catalog that
still offers one turns a settings dialog into a dead end. This module asks
every catalog entry whether its provider will still serve it, and reports the
ones that will not so they can be removed.

The check never raises on a provider error: a retired model is a finding, not
a crash, and one dead entry must not hide the state of the rest. The API key
comes from the environment, never from a command line, and never appears in
the output.
"""

import pytest

from services.accounts import chat_settings_service as catalog
from services.accounts.model_availability import (
    AvailabilityReport,
    check_catalog,
    format_report,
    provider_api_key_env,
)


class _Verified:
    """What the real provider returns from the factory."""

    def verify(self):
        """Run the one-token round trip the check makes."""


class _Providers:
    """A stand-in for the provider factory that fails for chosen models."""

    def __init__(self, unavailable=()):
        self.unavailable = set(unavailable)
        self.checked = []

    def __call__(self, credentials, client=None, use_cache=True):
        self.checked.append((credentials.provider, credentials.model))
        if credentials.model in self.unavailable:
            raise RuntimeError("The provider could not use this model.")
        return _Verified()


class TestKeyLookup:
    """TestKeyLookup."""

    def test_each_provider_reads_its_own_variable(self):
        """Do test each provider reads its own variable."""
        assert provider_api_key_env("google") == "PAPERMIND_CHECK_GOOGLE_API_KEY"
        assert provider_api_key_env("groq") == "PAPERMIND_CHECK_GROQ_API_KEY"

    def test_an_unknown_provider_has_no_variable(self):
        """Do test an unknown provider has no variable."""
        assert provider_api_key_env("acme") is None


class TestCheckCatalog:
    """TestCheckCatalog."""

    def test_every_catalog_entry_is_checked(self):
        """Do test every catalog entry is checked."""
        providers = _Providers()

        report = check_catalog(
            keys={"google": "g-key", "groq": "q-key"}, provider_factory=providers
        )

        assert sorted(report.checked) == sorted(
            (model.provider, model.id) for model in catalog.MODEL_CATALOG
        )

    def test_all_available_is_a_pass(self):
        """Do test all available is a pass."""
        report = check_catalog(
            keys={"google": "g-key", "groq": "q-key"},
            provider_factory=_Providers(),
        )

        assert report.ok is True
        assert report.unavailable == ()

    def test_a_retired_model_is_reported_not_raised(self):
        """Do test a retired model is reported not raised."""
        providers = _Providers(unavailable=["gemini-2.5-flash"])

        report = check_catalog(
            keys={"google": "g-key", "groq": "q-key"}, provider_factory=providers
        )

        assert report.ok is False
        assert report.unavailable == (("google", "gemini-2.5-flash"),)
        # The rest of the catalog was still checked: one dead entry must not
        # hide the state of the others.
        assert len(providers.checked) == len(catalog.MODEL_CATALOG)

    def test_a_missing_key_is_a_finding_naming_the_variable(self):
        """Do test a missing key is a finding naming the variable."""
        report = check_catalog(keys={"google": "g-key"}, provider_factory=_Providers())

        assert report.ok is False
        assert "PAPERMIND_CHECK_GROQ_API_KEY" in report.missing_keys

    def test_a_missing_key_does_not_pretend_to_have_checked_its_models(self):
        """Without a key there is nothing to ask the provider."""
        providers = _Providers()

        check_catalog(keys={"google": "g-key"}, provider_factory=providers)

        assert all(provider == "google" for provider, _ in providers.checked)

    def test_the_key_is_never_echoed(self):
        """Do test the key is never echoed."""
        report = check_catalog(
            keys={"google": "s3cret-google", "groq": "s3cret-groq"},
            provider_factory=_Providers(),
        )

        assert "s3cret-google" not in format_report(report)
        assert "s3cret-groq" not in format_report(report)


class TestFormatReport:
    """TestFormatReport."""

    def test_a_clean_run_says_so(self):
        """Do test a clean run says so."""
        report = check_catalog(
            keys={"google": "k", "groq": "k"}, provider_factory=_Providers()
        )

        assert format_report(report) == (
            f"All {len(catalog.MODEL_CATALOG)} catalog models are available."
        )

    def test_it_names_each_retired_entry(self):
        """Do test it names each retired entry."""
        report = check_catalog(
            keys={"google": "k", "groq": "k"},
            provider_factory=_Providers(unavailable=["openai/gpt-oss-20b"]),
        )

        text = format_report(report)

        assert "groq/openai/gpt-oss-20b" in text
        assert "MODEL_CATALOG" in text

    def test_it_names_each_variable_to_set(self):
        """Do test it names each variable to set."""
        report = check_catalog(keys={}, provider_factory=_Providers())

        assert "PAPERMIND_CHECK_GOOGLE_API_KEY" in format_report(report)


class TestReportShape:
    """TestReportShape."""

    def test_it_carries_how_many_entries_were_checked(self):
        """Do test it carries how many entries were checked."""
        report = check_catalog(
            keys={"google": "k", "groq": "k"}, provider_factory=_Providers()
        )

        assert isinstance(report, AvailabilityReport)
        assert report.checked_count == len(catalog.MODEL_CATALOG)


class TestWhatRetirementAsksFor:
    """What a person does with an unavailable entry."""

    def test_it_names_the_file_and_the_entry_to_delete(self):
        """Do test it names the file and the entry to delete."""
        report = check_catalog(
            keys={"google": "k", "groq": "k"},
            provider_factory=_Providers(unavailable=["gemini-3.5-flash"]),
        )

        printed = format_report(report)

        assert "MODEL_CATALOG" in printed
        assert "chat_settings_service.py" in printed
        assert "gemini-3.5-flash" in printed

    def test_it_warns_that_an_outage_looks_identical(self):
        """
        The other explanation is named too.

        The report is what a person reads before deleting a catalogue entry, so
        it has to name the reason they would otherwise skip the deletion.
        """
        report = check_catalog(
            keys={"google": "k", "groq": "k"},
            provider_factory=_Providers(unavailable=["gemini-3.5-flash"]),
        )

        assert "unreachable" in format_report(report)
