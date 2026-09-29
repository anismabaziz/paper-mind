"""
Turning dependency-audit tool output into findings, and findings into a verdict.

An advisory database entry is not automatically a problem for this project. A
critique in a package nothing here imports, or a denial of service in a parser
that only ever sees documents a user chose to upload, are different decisions
and deserve different treatment. What the audit owes the reader is that
distinction: a direct dependency carrying a new advisory blocks, and anything
else is reported with the reason it is not blocking.

Two tools, two report shapes:

- ``pip-audit --format=json`` reports installed packages, each with the
  advisories that name it. It does not say whether a package is a direct
  dependency, so that comes from the project's own ``pyproject.toml``.
- ``npm audit --format=json`` reports per-package entries whose ``via`` list
  mixes a package's own advisories (strings) with ones inherited from its
  dependencies (objects), and does say which packages are direct.

The parsers read the recorded text, not the tools, so both are checkable
without a network and without the tools installed. A report that cannot be
read raises: a check that reports "clean" because it misread its input is the
one failure mode worse than a vulnerable dependency.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The separator pip normalizes away when it compares distribution names
#: (PEP 503), so `PyJWT`, `pyjwt`, and `py_jwt` are one package.
_NAME_SEPARATORS = re.compile(r"[-_.]+")

#: A requirement's name, with any extras and version specifier removed.
_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")

#: An advisory identifier as GHSA and PyPA both spell it.
_ADVISORY_ID = re.compile(
    r"\b(?:GHSA|pypi)-[0-9a-zA-Z]{4}-[0-9a-zA-Z]{4}-[0-9a-zA-Z]{4}\b"
)


def normalize_name(name: str) -> str:
    """Return a distribution name the way pip compares it."""
    return _NAME_SEPARATORS.sub("-", name.strip()).lower()


@dataclass(frozen=True)
class Finding:
    """One advisory against one installed package."""

    ecosystem: str
    package: str
    version: str
    advisory: str
    severity: str
    direct: bool
    fix_available: bool

    def describe(self) -> str:
        """Return one line naming the package, the version, and the advisory."""
        fix = "fix available" if self.fix_available else "no fix published"
        return (
            f"{self.ecosystem}: {self.package} {self.version} — "
            f"{self.advisory} ({self.severity}, {fix})"
        )


@dataclass(frozen=True)
class Verdict:
    """What the findings mean for this project."""

    blocking: tuple[Finding, ...] = ()
    reported: tuple[Finding, ...] = ()
    accepted: tuple[Finding, ...] = ()

    @property
    def ok(self) -> bool:
        """Return True when nothing blocks the build."""
        return not self.blocking


def direct_python_packages(pyproject_path: Path) -> set[str]:
    """
    Return the packages this project asks for by name.

    Every declared group counts as direct: a development dependency is a
    choice someone made and reviewed, so an advisory against one is as much a
    decision as an advisory against a runtime one.
    """
    with Path(pyproject_path).open("rb") as handle:
        document = tomllib.load(handle)

    names: set[str] = set()
    for requirement in document.get("project", {}).get("dependencies", []):
        names.add(_requirement_name(requirement))
    for group in document.get("project", {}).get("optional-dependencies", {}).values():
        for requirement in group:
            names.add(_requirement_name(requirement))
    for group in document.get("dependency-groups", {}).values():
        for requirement in group:
            names.add(_requirement_name(requirement))
    return names


def _requirement_name(requirement: str) -> str:
    """Return the distribution name in a requirement string, extras removed."""
    match = _REQUIREMENT_NAME.match(requirement)
    if not match:
        raise ValueError(f"Cannot read a package name from {requirement!r}")
    return normalize_name(match.group(1))


def parse_pip_audit(report: Any, *, direct: set[str]) -> tuple[Finding, ...]:
    """
    Return the advisories in a ``pip-audit --format=json`` report.

    ``direct`` names the packages the project depends on directly; pip-audit
    does not know that, so it is supplied from ``pyproject.toml``.
    """
    dependencies = _report_entries(report, "pip-audit", "dependencies")
    findings: list[Finding] = []
    for dependency in dependencies:
        package = normalize_name(str(dependency.get("name", "")))
        if not package:
            continue
        for advisory in dependency.get("vulns") or []:
            findings.append(
                Finding(
                    ecosystem="python",
                    package=package,
                    version=str(dependency.get("version", "")),
                    advisory=str(advisory.get("id", "")),
                    # pip-audit's JSON carries no severity; the advisory
                    # identifier is what a reader looks up.
                    severity="unknown",
                    direct=package in direct,
                    fix_available=bool(advisory.get("fix_versions")),
                )
            )
    return tuple(findings)


def parse_npm_audit(report: Any) -> tuple[Finding, ...]:
    """
    Return the advisories in an ``npm audit --format=json`` report.

    npm counts one vulnerability per package, not per advisory, so a package
    with several is reported once for each: two different advisories against
    one package are two different risks, and collapsing them would hide the
    second.
    """
    vulnerabilities = _report_entries(report, "npm audit", "vulnerabilities")
    findings: list[Finding] = []
    for entry in vulnerabilities:
        package = str(entry.get("name", ""))
        severity = str(entry.get("severity", "unknown"))
        is_direct = bool(entry.get("isDirect"))
        fix_available = entry.get("fixAvailable") not in (False, None)
        for via in entry.get("via") or []:
            if isinstance(via, str):
                advisory, inherited_severity = via, severity
            elif isinstance(via, dict):
                inherited_severity = str(via.get("severity", severity))
                advisory = str(via.get("title") or via.get("url") or "")
                # npm puts the identifier in the advisory URL and only the
                # prose in the title, and the identifier is what a reader
                # looks up and what an acceptance is recorded against.
                identifier = _ADVISORY_ID.search(f"{via.get('url', '')} {advisory}")
            else:
                continue
            if isinstance(via, str) or identifier is None:
                identifier = _ADVISORY_ID.search(advisory)
            findings.append(
                Finding(
                    ecosystem="node",
                    package=package,
                    version="",
                    advisory=identifier.group(0) if identifier else advisory,
                    severity=inherited_severity,
                    # An advisory inherited through a dependency stays
                    # attributed to the direct package, because that is the
                    # one this project chose and can upgrade.
                    direct=is_direct,
                    fix_available=fix_available,
                )
            )
    return tuple(findings)


def _report_entries(report: Any, tool: str, key: str) -> list[dict[str, Any]]:
    """Return a report's package entries, or raise naming what went unread."""
    value = report.get(key) if isinstance(report, dict) else None
    # pip-audit keys its packages by list, npm by object; both are read here so
    # neither tool's shape is a special case at the call site.
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return list(value.values())
    raise ValueError(
        f"Cannot read the {tool} report: expected a JSON object with "
        f"{key!r}. A report this audit cannot read is not a clean report."
    )


def decide(
    findings: tuple[Finding, ...], *, accepted: dict[str, str] | None = None
) -> Verdict:
    """
    Split findings into the ones that block and the ones that only get reported.

    A direct dependency with a new advisory blocks, because the fix is a
    version bump in a file this project owns. An advisory that has been
    accepted keeps its reason attached: silence would read as "reviewed and
    fine", and a bare exception list loses the reasoning six months later.
    """
    reasons = accepted or {}
    blocking: list[Finding] = []
    reported: list[Finding] = []
    accepted_findings: list[Finding] = []
    for finding in findings:
        if finding.advisory in reasons:
            accepted_findings.append(finding)
        elif finding.direct:
            blocking.append(finding)
        else:
            reported.append(finding)
    return Verdict(
        blocking=tuple(blocking),
        reported=tuple(reported),
        accepted=tuple(accepted_findings),
    )
