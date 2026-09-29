"""
Asking each catalog model whether its provider still serves it.

A catalog entry is a promise that a user can pick that model and get an
answer. Providers retire models on their own schedule, and a shipped catalog
that still offers one turns the settings dialog into a dead end — the failure
only shows up after a user has chosen a provider, pasted a key, and waited.

This module asks the provider about every entry and reports the ones it will
not serve, so a scheduled job can fail while there is still time to remove
them. A retired model is a finding, not a crash: one dead entry must not stop
the remaining entries from being checked, and must not mask whether they work.

API keys are read from the environment and never from a command line, because
a command line is visible to every process on the machine and lands in shell
history. The report records which variables were missing, never their values.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from services.accounts.chat_settings_service import (
    MODEL_CATALOG,
    verification_error_message,
)
from services.llm.base import ChatCredentials
from services.llm.factory import build_chat_provider

#: The environment variable each provider's key is read from. One per provider
#: so a check for a single provider needs only that provider's key.
PROVIDER_API_KEY_ENV = {
    "google": "PAPERMIND_CHECK_GOOGLE_API_KEY",
    "groq": "PAPERMIND_CHECK_GROQ_API_KEY",
}


def provider_api_key_env(provider: str) -> str | None:
    """Return the variable a provider's key is read from, if it has one."""
    return PROVIDER_API_KEY_ENV.get(provider)


@dataclass(frozen=True)
class AvailabilityReport:
    """What the providers said about every catalog entry."""

    checked: tuple[tuple[str, str], ...] = ()
    unavailable: tuple[tuple[str, str], ...] = ()
    missing_keys: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    checked_count: int = 0

    @property
    def ok(self) -> bool:
        """Return True when every catalog entry was checked and answered."""
        return not self.unavailable and not self.missing_keys


def check_catalog(
    keys: dict[str, str],
    *,
    provider_factory: Callable[..., Any] = build_chat_provider,
    catalog: tuple[Any, ...] = MODEL_CATALOG,
) -> AvailabilityReport:
    """
    Ask each provider whether it still serves every model in ``catalog``.

    ``keys`` maps a provider name to the API key to use. A provider with no
    key is reported as a missing key and its models are left unchecked,
    because a check that cannot run must not be recorded as a pass. Errors
    from the provider become findings; they never stop the sweep.
    """
    checked: list[tuple[str, str]] = []
    unavailable: list[tuple[str, str]] = []
    missing: list[str] = []
    reasons: list[str] = []
    seen_providers: set[str] = set()

    for model in catalog:
        key = (keys.get(model.provider) or "").strip()
        if not key:
            if model.provider not in seen_providers:
                seen_providers.add(model.provider)
                variable = provider_api_key_env(model.provider)
                missing.append(
                    variable or f"an API key for {model.provider} (no variable is set)"
                )
            continue
        checked.append((model.provider, model.id))
        try:
            provider = provider_factory(
                ChatCredentials(
                    provider=model.provider,
                    model=model.id,
                    api_key=key,
                    budget=model.chat_budget(),
                ),
                use_cache=False,
            )
            provider.verify()
        except Exception as exc:  # any provider error is a finding, not a crash
            unavailable.append((model.provider, model.id))
            reasons.append(
                f"{model.provider}/{model.id}: {verification_error_message(exc)}"
            )

    return AvailabilityReport(
        checked=tuple(checked),
        unavailable=tuple(unavailable),
        missing_keys=tuple(missing),
        reasons=tuple(reasons),
        checked_count=len(checked),
    )


def format_report(report: AvailabilityReport) -> str:
    """
    Render a report as the lines a job log and a human both need.

    A retired entry is named exactly, so removing it is a one-line deletion in
    ``MODEL_CATALOG``. It is not removed automatically, and that is deliberate:
    a provider that is briefly unreachable or rate-limiting answers exactly
    like one that has retired a model, and a job that edited the catalogue in
    response would strip it on an outage. The judgement — retire it, or is this
    the provider having a bad day — belongs to whoever reads the report, and
    putting it in a reviewed commit is what makes it a decision rather than an
    accident.
    """
    if report.ok:
        return f"All {report.checked_count} catalog models are available."

    lines: list[str] = []
    if report.missing_keys:
        lines.append("Not checked, no API key in the environment:")
        lines.extend(f"  - set {variable}" for variable in report.missing_keys)
    if report.unavailable:
        lines.append(
            f"Unavailable ({len(report.unavailable)} of {report.checked_count} checked):"
        )
        lines.extend(f"  - {reason}" for reason in report.reasons)
        lines.append(
            "Delete each entry above from MODEL_CATALOG in "
            "services/accounts/chat_settings_service.py, unless the provider is "
            "merely unreachable right now."
        )
    return "\n".join(lines)


def keys_from_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Return the provider keys present in the environment, by provider."""
    source = os.environ if environ is None else environ
    return {
        provider: source[variable]
        for provider, variable in PROVIDER_API_KEY_ENV.items()
        if source.get(variable, "").strip()
    }
