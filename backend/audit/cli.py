"""
The audit command CI runs.

Two checks, kept apart on purpose because they fail for different reasons and
are fixed by different people:

- ``python`` and ``node`` run the ecosystem's own audit tool and decide
  whether the findings block. These are deterministic: given the same
  lockfiles and the same advisory database, the answer does not change, which
  is what makes them safe to gate a pull request on.
- ``models`` asks the model providers whether every catalog entry still works,
  and checks that the pinned local revisions still exist upstream. This needs
  the network, the provider keys, and hours rather than seconds, so it belongs
  in a scheduled job whose failure is a signal rather than a broken build.

Keys are read from the environment and never from an argument, because an
argument is visible to every process on the machine and lands in shell history.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from audit.policy import (
    Verdict,
    decide,
    direct_python_packages,
    parse_npm_audit,
    parse_pip_audit,
)
from audit.report import format_report

#: Recorded risk decisions, keyed by advisory identifier.
ACCEPTED_PATH = Path(__file__).with_name("accepted.json")

#: The folder holding ``backend/`` and ``frontend/``. Both audits are run
#: against a checkout, so both resolve their lockfiles from here.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def load_accepted(path: Path = ACCEPTED_PATH) -> dict[str, str]:
    """
    Return the recorded risk decisions, or an empty map when there are none.

    The ``accepted`` key is required rather than optional. A file without it
    would otherwise parse as a map of prose keys, and a comment whose text
    happened to name an advisory would silently silence that advisory.
    """
    if not path.exists():
        return {}
    document = json.loads(path.read_text())
    accepted = document.get("accepted") if isinstance(document, dict) else None
    if not isinstance(accepted, dict):
        raise ValueError(
            f"{path} must hold an {{'accepted': {{<advisory id>: <reason>}}}} object"
        )
    return {str(key): str(reason) for key, reason in accepted.items()}


def _run_json(command: list[str], cwd: Path, tool: str) -> Any:
    """
    Run an audit tool and return its JSON report.

    Both tools report findings through their exit code — ``npm audit`` exits
    non-zero on a clean tree as well, because it treats "nothing to fix" as a
    failure of the check rather than of the tree. So the exit code is
    recorded and the report is what decides, not the code. Reading stdout is
    the only way to tell a finding from a crash.

    Empty or unparseable output raises: a report the audit cannot read is not a
    clean report, and returning an empty finding list would turn a broken tool
    into a green job.
    """
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=False
    )
    if not result.stdout.strip():
        raise ValueError(
            f"{tool} produced no report (exit {result.returncode}): "
            f"{result.stderr.strip() or 'no output'}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{tool} produced output that is not JSON (exit {result.returncode}): "
            f"{result.stdout[:200]}"
        ) from error


def audit_python(root: Path, accepted: dict[str, str]) -> Verdict:
    """Audit the locked Python environment and decide whether it blocks."""
    report = _run_json(
        [
            "uv",
            "run",
            "--with",
            "pip-audit",
            "pip-audit",
            "--local",
            "--format=json",
            "--progress-spinner=off",
        ],
        cwd=root,
        tool="pip-audit",
    )
    direct = direct_python_packages(root / "pyproject.toml")
    return decide(parse_pip_audit(report, direct=direct), accepted=accepted)


def audit_node(root: Path, accepted: dict[str, str]) -> Verdict:
    """Audit the locked Node tree and decide whether it blocks."""
    report = _run_json(["npm", "audit", "--json"], cwd=root, tool="npm audit")
    return decide(parse_npm_audit(report), accepted=accepted)


def check_pinned_revisions() -> tuple[bool, str]:
    """
    Check that every pinned local model revision still exists upstream.

    A pin that no longer resolves is worse than no pin: the application
    downloads whatever the repository serves now, or fails at first use, and
    either way the recorded identity is fiction.
    """
    from services.models import PINNED_MODEL_REVISIONS, canonical_model_id

    lines: list[str] = []
    failed = False
    for model_id, revision in sorted(PINNED_MODEL_REVISIONS.items()):
        try:
            payload = _run_json(
                [
                    "curl",
                    "--silent",
                    "--show-error",
                    "--fail",
                    "--max-time",
                    "30",
                    "https://huggingface.co/api/models/"
                    f"{canonical_model_id(model_id)}/revision/{revision}",
                ],
                cwd=Path("."),
                tool=f"the model hub (checking {model_id})",
            )
        except Exception as exc:
            # Any failure here is a finding, not a crash: one unreachable
            # repository must not stop the rest from being checked.
            failed = True
            lines.append(f"  - {model_id} @ {revision[:12]}: {exc}")
            continue
        served = payload.get("sha", "")
        if served != revision:
            failed = True
            lines.append(
                f"  - {model_id} @ {revision[:12]}: the repository now serves {served[:12]}"
            )
    if failed:
        return False, "Pinned model revisions that no longer resolve:\n" + "\n".join(
            lines
        )
    return True, f"All {len(PINNED_MODEL_REVISIONS)} pinned model revisions resolve."


def report_missing_comparison(changed_csv: str) -> None:
    """
    Say which of the changed files ask for an evaluation comparison.

    Prints nothing when the change cannot move a measurement, so the check is
    quiet on the overwhelming majority of pull requests. It does not fail the
    build: requiring a paid live run before a security patch can merge is the
    wrong trade, and the omission is caught by review rather than by a red
    check that gets force-pushed past.
    """
    from audit.comparison import paths_needing_a_comparison

    needing = paths_needing_a_comparison(
        {path.strip() for path in changed_csv.split(",") if path.strip()}
    )
    if not needing:
        return
    print("This change can move a published measurement without failing a test:")
    for path in needing:
        print(f"  - {path}")
    print(
        "Run the evaluator against the reported half of the case set and attach "
        "the comparison, or say in the pull request that the numbers did not "
        "move. From backend/:"
    )
    print(
        "  uv run python -m evaluation.cli --live --split validation \\\n"
        "    --report reports/<new-name> --compare-report "
        "reports/2026-09-retrieval-baseline-v1"
    )


def _node_audit(root: Path, accepted: dict[str, str]) -> Verdict:
    """Audit the locked Node tree, resolved from the repository root."""
    return audit_node(root / "frontend", accepted)


def _python_audit(root: Path, accepted: dict[str, str]) -> Verdict:
    """Audit the locked Python environment, resolved from the repository root."""
    return audit_python(root / "backend", accepted)


def main(argv: list[str] | None = None) -> int:
    """Run the requested audit and exit non-zero when it blocks."""
    parser = argparse.ArgumentParser(
        prog="python -m audit.cli",
        description="Audit dependencies and pinned model revisions.",
    )
    parser.add_argument(
        "check",
        choices=("python", "node", "models", "revisions", "comparison", "all"),
        help="Which audit to run.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="Repository root (the folder holding backend/ and frontend/).",
    )
    parser.add_argument(
        "--changed",
        default="",
        help=(
            "Comma-separated repository-relative paths this change touches, "
            "for the comparison check."
        ),
    )
    args = parser.parse_args(argv)

    accepted = load_accepted()
    failed = False

    if args.check in ("comparison",):
        report_missing_comparison(args.changed)
    if args.check in ("python", "all"):
        failed |= _section("Python dependencies", accepted, _python_audit, args.root)
    if args.check in ("node", "all"):
        failed |= _section("Node dependencies", accepted, _node_audit, args.root)

    if args.check in ("revisions", "all"):
        try:
            ok, text = check_pinned_revisions()
        except Exception as error:  # reported, never raised
            ok, text = False, f"Could not complete: {error}"
        print("Pinned model revisions\n" + text)
        failed = failed or not ok

    if args.check in ("models", "all"):
        # Imported here, not at module scope: this sweep needs the provider
        # SDKs, while the revision check below needs nothing installed. Making
        # it lazy lets that check run on a bare interpreter, which is what
        # keeps its CI job to three HTTP calls and no dependency install.
        from services.accounts.model_availability import (
            check_catalog,
            format_report as format_availability,
            keys_from_env,
        )

        report = check_catalog(keys_from_env())
        print("Model catalog\n" + format_availability(report))
        failed = failed or not report.ok

    return 1 if failed else 0


def _section(title: str, accepted: dict[str, str], audit, root: Path) -> bool:
    """
    Print one ecosystem's audit and return whether it blocks.

    A tool that cannot produce a report is reported as such and counts as a
    failure. Letting it propagate would make the job log a traceback, which
    says less about what went wrong than the message does, and a traceback is
    easy to misread as the audit's own bug.
    """
    try:
        verdict = audit(root, accepted)
    except Exception as error:  # reported, never raised
        print(f"{title}\nCould not complete: {error}")
        return True
    print(f"{title}\n" + format_report(verdict, accepted=accepted))
    return not verdict.ok


if __name__ == "__main__":
    sys.exit(main())
