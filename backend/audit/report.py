"""
Rendering an audit verdict for a job log and for a reviewer.

The report is read in two situations with different needs: a person skimming
a failed job, who needs to know what to do next, and a person auditing a
decision later, who needs the reason an advisory was allowed to stay. Both
are served by listing the findings, listing the decisions, and saying plainly
what passed.
"""

from __future__ import annotations

from collections.abc import Mapping

from audit.policy import Verdict


def format_report(
    verdict: Verdict, *, accepted: Mapping[str, str] | None = None
) -> str:
    """Return the lines a job log and a reviewer both need."""
    reasons = accepted or {}
    if verdict.ok and not verdict.reported and not verdict.accepted and not reasons:
        return "No known vulnerabilities in the audited dependencies."

    lines: list[str] = []
    if verdict.blocking:
        lines.append(
            f"{len(verdict.blocking)} vulnerable direct dependencies. "
            "Update them, or record a decision in audit/accepted.json."
        )
        lines.extend(f"  - {finding.describe()}" for finding in verdict.blocking)
    if verdict.reported:
        lines.append(
            f"{len(verdict.reported)} advisories in transitive dependencies "
            "(reported, not blocking):"
        )
        lines.extend(f"  - {finding.describe()}" for finding in verdict.reported)
    if verdict.accepted:
        lines.append(f"{len(verdict.accepted)} accepted advisories:")
        for finding in verdict.accepted:
            lines.append(
                f"  - {finding.describe()}\n"
                f"    accepted: {reasons.get(finding.advisory, 'no reason recorded')}"
            )
    # An acceptance with nothing behind it is a decision nobody has revisited
    # since the package was upgraded. It is not a failure, but a reader should
    # not have to diff the file to notice.
    stale = sorted(set(reasons) - {f.advisory for f in verdict.accepted})
    if stale:
        lines.append(
            f"{len(stale)} recorded acceptances no longer match any finding; "
            "remove them from audit/accepted.json:"
        )
        lines.extend(f"  - {advisory}" for advisory in stale)
    if verdict.ok:
        lines.append("Nothing blocking.")
    return "\n".join(lines)
