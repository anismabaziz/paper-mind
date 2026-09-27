"""
A checked-in report: what was measured, on what, and by which models.

A retrieval number on its own is a claim. What a reviewer can check is the
number together with everything that produced it: the revision, the version of
the case set, the hash of every document, the index manifest, the prompts, the
models and their revisions, the settings, and the environment. Those travel in
a manifest, and a report is the directory holding the manifest, one file per
experiment, and a summary a person reads first.

The report also says what it did *not* measure. A retrieval-only run has no
answer, citation, abstention, token, or dollar figure, and a summary that left
those cells empty would read as a run that measured them and got nothing, so
each kind of evidence is either measured somewhere in the report or carries the
reason it was not.

Reproducibility is checked, not asserted. Timing is a property of the machine
and the hour, so it is deliberately outside the manifest's digest: two runs that
measured the same things reproduce the manifest, and a run that measured
something different, or with a different model revision, says which field moved.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation import experiments
from evaluation.dataset import TUNING
from evaluation.experiments import describe
from services.answering import AnswerSettings
from services.citations import PROMPT_VERSION
from services.indexing.manifest import COLLECTION_SCHEMA_VERSION
from services.retrieval.hybrid import RRF_K, TOKENIZER_VERSION
from services.retrieval.vector_service import FETCH_K, MAX_RETRIEVED_SOURCES

#: Bumped when the shape of a report changes, so an old directory is not read
#: as a current one.
REPORT_VERSION = "1"

#: What a retrieval-only measurement has and has not measured. The four kinds
#: the answer path grades are stated here too, rather than left out: a report
#: that omitted them would read as a run that decided nothing at all.
RETRIEVAL_ONLY_REASON = "retrieval-only measurement, nothing asked a model"

#: Fields of a result that are measurements of the machine rather than of the
#: application, and so are left out of the digest a reproduction is judged on.
_TIMING_FIELDS = frozenset(
    {
        "latency",
        "comparison",
        "seconds",
        "indexed_seconds",
        "cold_start",
        "steady_state",
    }
)

#: The kinds of evidence a report distinguishes. Deterministic checks are
#: computed from text; model-graded results come from a judge; human
#: calibration compares the judge with a person; provider failures are the runs
#: that never reached a model at all.
EVIDENCE = (
    "deterministic",
    "model_graded",
    "human_calibration",
    "provider_failures",
    "answers",
    "citations",
    "abstention",
    "tokens",
    "cost",
)

#: The answer metrics a deterministic grader decides, as opposed to the two a
#: model judge decides.
DETERMINISTIC_METRICS = ("citation_precision", "citation_recall", "abstention")

MANIFEST_NAME = "manifest.json"
RESULTS_DIR = "results"
SUMMARY_NAME = "README.md"


def _package_version(name: str) -> str:
    """Return an installed package's version, or a note that it is unknown."""
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001 - a missing package is worth reporting
        return "not installed"


def git_revision(root: Path | None = None) -> dict[str, Any]:
    """
    Return the revision the report was measured on.

    A report measured on an uncommitted tree says so: the numbers exist, but the
    code that produced them does not, which is a different thing from a
    revision that can be checked out and re-run.
    """
    root = Path(root or Path(__file__).parent)
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        subject = subprocess.run(
            ["git", "log", "-1", "--pretty=%s"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"revision": "", "dirty": False, "subject": f"unavailable: {exc}"}
    return {
        "revision": revision,
        "dirty": bool(status),
        "subject": subject,
        "files_changed": len(status.splitlines()) if status else 0,
    }


def environment_fingerprint() -> dict[str, Any]:
    """
    Return the environment a report was measured in.

    Deliberately a list rather than a dump of the machine's environment: an
    evaluation run holds two provider keys, and a report is committed. The
    version of a retrieval implementation is here because a change in the
    tokenizer, the fusion constant, or the candidate depth moves the numbers
    without any file in the application changing.
    """
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            name: _package_version(name)
            for name in (
                "qdrant-client",
                "sentence-transformers",
                "sqlalchemy",
                "tiktoken",
            )
        },
        "retrieval": {
            "rrf_k": RRF_K,
            "sparse_tokenizer": TOKENIZER_VERSION,
            "candidate_depth": FETCH_K,
            "context_sources": MAX_RETRIEVED_SOURCES,
            "collection_schema": COLLECTION_SCHEMA_VERSION,
        },
    }


def evidence_from_run(run: dict[str, Any]) -> dict[str, Any]:
    """
    Return what one answer-path run measured, and what it could not decide.

    A provider that died is counted as a failure rather than as an answer with
    no good citations, which is why the answer and token sections are not
    measured for a run in which nothing was written.
    """
    answers = run["answers"]
    cost = run["cost"]
    outcomes = run["outcomes"]
    calibration = run.get("calibration")
    judged = [case for case in run["cases"] if case.get("verdict")]
    failures = outcomes.get("provider_error", 0) + outcomes.get("context_fallback", 0)
    priced = cost["priced_cases"] > 0
    return {
        "deterministic": {
            "measured": True,
            "graded": sum(answers[name]["graded"] for name in DETERMINISTIC_METRICS),
        },
        "model_graded": (
            {"measured": True, "verdicts": len(judged)}
            if run.get("judge")
            else {"measured": False, "reason": "no judge was configured for this run"}
        ),
        "human_calibration": (
            {"measured": True, "agreement": calibration["agreement"]}
            if calibration
            else {"measured": False, "reason": "the calibration set was not run"}
        ),
        "provider_failures": {"measured": True, "failures": failures},
        "answers": (
            {"measured": True, "generated": answers["generated"]}
            if answers["generated"]
            else {"measured": False, "reason": "no model wrote an answer"}
        ),
        "citations": (
            {"measured": True, "graded": answers["citation_precision"]["graded"]}
            if answers["citation_precision"]["graded"]
            else {"measured": False, "reason": "no answer carried a citation"}
        ),
        "abstention": (
            {"measured": True, "graded": answers["abstention"]["graded"]}
            if answers["abstention"]["graded"]
            else {"measured": False, "reason": "no case required an abstention"}
        ),
        "tokens": (
            {
                "measured": True,
                "input": cost["input_tokens"],
                "output": cost["output_tokens"],
            }
            if priced
            else {"measured": False, "reason": "no case reached a model"}
        ),
        "cost": (
            {"measured": True, "usd": cost["usd"]}
            if priced
            else {"measured": False, "reason": "no case reached a model"}
        ),
    }


def _canonical(value: Any) -> str:
    """Return one value as JSON, in a form that does not depend on key order."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def measurements_digest(results: Sequence[dict[str, Any]]) -> str:
    """
    Return a hash over what the results measured, timings left out.

    Latency and dollars move on every run, on any machine, so a report that
    digested them could never be said to have reproduced. What is left is the
    part a second run has to match: the retrieval numbers, the outcomes, the
    verdicts, and the evidence each run recorded.
    """
    return hashlib.sha256(
        _canonical([_without_timings(result) for result in results]).encode()
    ).hexdigest()


def _without_timings(value: Any) -> Any:
    """
    Return one result with every timing left out, wherever it is nested.

    Indexing seconds sit inside a document's index state rather than beside it,
    so pruning only the top level would let a re-index of the same documents
    read as a different measurement.
    """
    if isinstance(value, dict):
        return {
            name: _without_timings(nested)
            for name, nested in value.items()
            if name not in _TIMING_FIELDS
        }
    if isinstance(value, list):
        return [_without_timings(item) for item in value]
    return value


def evidence_for_retrieval(questions: int) -> dict[str, Any]:
    """
    Return what a retrieval-only measurement measured, for every evidence kind.

    The four kinds the answer path grades are stated here too, rather than left
    out: a report that omitted them would read as a run that decided nothing at
    all, and a provider that was never called cannot have failed.
    """
    unmeasured = {"measured": False, "reason": RETRIEVAL_ONLY_REASON}
    return {
        "deterministic": {"measured": True, "graded": questions},
        "model_graded": unmeasured,
        "human_calibration": unmeasured,
        "provider_failures": {
            "measured": False,
            "reason": "no provider was called, so no call could fail",
        },
        "answers": unmeasured,
        "citations": unmeasured,
        "abstention": unmeasured,
        "tokens": unmeasured,
        "cost": unmeasured,
    }


def _split_of(results: Sequence[dict[str, Any]]) -> str:
    """Return the one split every result was measured on, or refuse the set."""
    splits = {result.get("split") for result in results}
    if len(splits) != 1:
        raise ValueError(
            "a report compares results measured on one split; "
            f"got {sorted(str(split) for split in splits)}"
        )
    return splits.pop() or ""


def _model(settings, run_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return the models a report names: the two local ones and the two remote."""
    run = run_records[0] if run_records else {}
    judge = run.get("judge")
    return {
        "embedding": {
            "id": settings.embedding.embedding_model,
            "revision": settings.embedding.revision,
        },
        "reranker": {
            "id": settings.rerank.rerank_model,
            "revision": settings.rerank.revision,
        },
        "generator": (
            {"provider": run["provider"], "id": run["model"]}
            if "provider" in run
            else None
        ),
        "judge": (
            {
                "provider": judge["provider"],
                "id": judge["model"],
                "rubric": judge["rubric_version"],
            }
            if judge
            else None
        ),
    }


def evidence_summary(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return, per kind of evidence, where it was measured and where it was not."""
    summary: dict[str, Any] = {}
    for kind in EVIDENCE:
        measured_in = []
        missing = []
        for result in results:
            stated = result.get("evidence", {}).get(kind)
            if stated is None:
                continue
            if stated.get("measured"):
                measured_in.append(result["experiment"])
            else:
                missing.append(
                    {
                        "experiment": result["experiment"],
                        "reason": stated.get("reason", "not measured"),
                    }
                )
        summary[kind] = {
            "measured_in": measured_in,
            "not_measured": missing,
            "measured": bool(measured_in),
        }
    return summary


def build_manifest(
    *,
    dataset,
    settings,
    results: Sequence[dict[str, Any]],
    revision: dict[str, Any] | None = None,
    run_records: Sequence[dict[str, Any]] = (),
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Return everything a reader needs before reading a number.

    The manifest holds no measurements of its own beyond the digest, so a second
    run of the same experiments is compared against the first by comparing two
    manifests rather than two directories of numbers.
    """
    run = run_records[0] if run_records else {}
    return {
        "report_version": REPORT_VERSION,
        "revision": revision or git_revision(),
        "dataset": {
            "version": dataset.version,
            "reviewed_on": dataset.reviewed_on,
            "split": _split_of(results),
            # The questions asked, not the sum over the columns: the same
            # questions are asked of every experiment, and adding them up would
            # read as a set eleven times larger than the one that was reviewed.
            "cases": len(
                {case["id"] for result in results for case in result.get("cases", [])}
            ),
            # The cases of this split a reported run leaves out, named here
            # rather than dropped quietly.
            "held_back": sorted(
                {name for result in results for name in result.get("held_back", [])}
            ),
        },
        "documents": [document.to_dict() for document in dataset.documents],
        "prompts": {
            "citation": PROMPT_VERSION,
            "judge_rubric": (run.get("judge") or {}).get("rubric_version"),
        },
        "models": _model(settings, run_records),
        "settings": AnswerSettings.from_settings(settings).to_dict(),
        "retrieval_methods": run.get("retrieval_methods", []),
        "environment": environment or environment_fingerprint(),
        "experiments": described_experiments(results),
        "evidence": evidence_summary(results),
        "results": [
            {
                "experiment": result["experiment"],
                "family": result.get("family", ""),
                "split": result.get("split", ""),
                "questions": result.get("retrieval", {}).get("questions", 0),
            }
            for result in results
        ],
        "measurements_digest": measurements_digest(results),
    }


def described_experiments(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Return the declared experiments a report's results were measured under.

    A result that names an experiment nobody declared is a result whose column
    in the table says nothing, so the registry is the only place a report can
    learn what a variant changed.
    """
    return describe(
        [experiments.experiment(result["experiment"]) for result in results]
    )


def _leaves(value: Any, path: str = "") -> dict[str, Any]:
    """Return a manifest's leaf values by dotted path."""
    if isinstance(value, dict):
        leaves: dict[str, Any] = {}
        for name, nested in value.items():
            leaves.update(_leaves(nested, f"{path}.{name}" if path else str(name)))
        return leaves
    return {path: value}


def manifest_changes(
    previous: dict[str, Any], current: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return every field that differs between two manifests, in path order."""
    before, after = _leaves(previous), _leaves(current)
    return [
        {"field": path, "before": before.get(path), "after": after.get(path)}
        for path in sorted(set(before) | set(after))
        if before.get(path) != after.get(path)
    ]


def reproduction(current: Path, previous: Path) -> dict[str, Any]:
    """
    Return whether a new report reproduces a published one, and what moved.

    A changed model revision is called out on its own, because that is the one
    difference a reader cannot fix by checking out the same revision: the model
    behind it moved without a line of this codebase changing.
    """
    published = load_manifest(previous)
    fresh = load_manifest(current)
    changes = manifest_changes(published, fresh)
    return {
        "reproduced": not changes,
        # Whether the numbers moved, kept apart from whether the setup that
        # produced them is the same: a different machine changes the environment
        # without changing a single measurement.
        "measurements_changed": (
            published["measurements_digest"] != fresh["measurements_digest"]
        ),
        "changes": changes,
        "model_revisions_changed": [
            change for change in changes if change["field"].startswith("models.")
        ],
    }


def load_manifest(directory: Path) -> dict[str, Any]:
    """Return the manifest a report directory was written with."""
    return json.loads((Path(directory) / MANIFEST_NAME).read_text(encoding="utf-8"))


def write_report(
    directory: Path, manifest: dict[str, Any], results: Sequence[dict[str, Any]]
) -> Path:
    """
    Write a report: the manifest, one file per experiment, and the summary.

    The summary is written from the same manifest and results as the files, so
    a table in it cannot drift from the numbers beside it.
    """
    directory = Path(directory)
    (directory / RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    for result in results:
        (directory / RESULTS_DIR / f"{result['experiment']}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (directory / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (directory / SUMMARY_NAME).write_text(render(manifest, results), encoding="utf-8")
    return directory


@dataclass(frozen=True)
class Column:
    """One experiment's numbers as the summary table shows them."""

    id: str
    family: str
    hit_rate: float | None
    recall: float | None
    mrr: float | None
    ndcg: float | None
    p50: float | None
    p95: float | None


def _number(value: Any, places: int = 3) -> str:
    """Return a number for a table cell, or a dash when there is none."""
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.{places}f}"
    return str(value)


def columns(results: Iterable[dict[str, Any]]) -> list[Column]:
    """Return one column per result, in the order the report lists them."""
    table = []
    for result in results:
        retrieval = result.get("retrieval", {})
        latency = result.get("latency", {}).get("retrieval_seconds", {})
        table.append(
            Column(
                id=result["experiment"],
                family=result.get("family", ""),
                hit_rate=retrieval.get("hit_rate"),
                recall=retrieval.get("recall"),
                mrr=retrieval.get("mrr"),
                ndcg=retrieval.get("ndcg"),
                p50=latency.get("p50"),
                p95=latency.get("p95"),
            )
        )
    return table


def _moved(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return every question that moved, with the move and the direction."""
    moved = []
    for result in results:
        comparison = result.get("comparison") or {}
        for direction in ("regressions", "improvements"):
            for row in comparison.get(direction, []):
                moved.append(
                    {
                        "experiment": result["experiment"],
                        "direction": direction,
                        "id": row["id"],
                        "delta": row.get("delta", {}).get("ndcg_at_k", 0.0),
                    }
                )
    return moved


def render(manifest: dict[str, Any], results: Sequence[dict[str, Any]]) -> str:
    """
    Return the report a reviewer reads first.

    It leads with what was measured and what was not, because a column of
    numbers is worth nothing without knowing whether the run behind it asked a
    model, and then with the comparison itself.
    """
    revision = manifest["revision"]
    dataset = manifest["dataset"]
    models = manifest["models"]
    environment = manifest["environment"]
    lines = [
        f"# Retrieval baseline: case set {dataset['version']}",
        "",
        f"Measured on revision `{revision['revision'][:12] or 'unknown'}` "
        f"({revision.get('subject', '')})"
        + (", with uncommitted changes" if revision.get("dirty") else ""),
        f"Case set {dataset['version']} (reviewed {dataset['reviewed_on']}), split "
        f"`{dataset['split']}`: {dataset['cases']} questions asked of each of the "
        f"{len(results)} experiments, "
        + (
            f"{len(dataset['held_back'])} of the split's cases held back "
            f"({', '.join(f'`{name}`' for name in dataset['held_back'])})."
            if dataset["held_back"]
            else "no cases held back."
        ),
        f"Embedding `{models['embedding']['id']}`"
        + (
            f" at `{models['embedding']['revision']}`"
            if models["embedding"]["revision"]
            else ""
        ),
        f"Reranker `{models['reranker']['id']}`"
        + (
            f" at `{models['reranker']['revision']}`"
            if models["reranker"]["revision"]
            else ""
        ),
        f"Citation prompt `{manifest['prompts']['citation']}`, Python "
        f"{environment['python']}, RRF k="
        f"{environment['retrieval']['rrf_k']}, candidate depth "
        f"{environment['retrieval']['candidate_depth']}.",
        "",
        "## What this run measured",
        "",
    ]
    for kind in EVIDENCE:
        stated = manifest["evidence"][kind]
        if stated["measured"]:
            lines.append(
                f"- **{kind.replace('_', ' ')}**: measured in "
                + ", ".join(f"`{name}`" for name in stated["measured_in"])
            )
        elif stated["not_measured"]:
            reasons = sorted({row["reason"] for row in stated["not_measured"]})
            lines.append(
                f"- **{kind.replace('_', ' ')}**: not measured — " + "; ".join(reasons)
            )
    lines += [
        "",
        "## Experiments",
        "",
        (
            "This is the tuning half: the split every variant's value was "
            "chosen on. A number here is a decision, not a result — the "
            "reported half is a separate run, and no variant's value was "
            "changed after reading it."
            if dataset["split"] == TUNING
            else "Each variant's value was chosen on the `tuning` split, never "
            "on the split reported here."
        ),
        "",
        "| Experiment | Family | Changes | Hit@k | Recall | MRR | nDCG | Retrieval p50 (s) | Retrieval p95 (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    declared = {item["id"]: item for item in manifest["experiments"]}
    for column in columns(results):
        changes = declared.get(column.id, {}).get("changes", {})
        changed = ", ".join(f"{name}={value}" for name, value in changes.items()) or "-"
        lines.append(
            f"| `{column.id}` | {column.family} | {changed} | "
            f"{_number(column.hit_rate)} | {_number(column.recall)} | "
            f"{_number(column.mrr)} | {_number(column.ndcg)} | "
            f"{_number(column.p50, 4)} | {_number(column.p95, 4)} |"
        )
    moved = _moved(results)
    lines += ["", "## Questions that moved", ""]
    if moved:
        lines += [
            "| Experiment | Direction | Question | nDCG change |",
            "| --- | --- | --- | --- |",
        ]
        lines += [
            f"| `{row['experiment']}` | {row['direction']} | `{row['id']}` | "
            f"{_number(row['delta'])} |"
            for row in moved
        ]
    else:
        lines.append("No question changed rank between the baseline and any variant.")
    lines += [
        "",
        "## Reproducing this report",
        "",
        "```",
        "docker compose up db qdrant",
        "uv run python -m evaluation.cli --live --report <directory> --ablate",
        "uv run python -m evaluation.cli --live --report <directory> --compare-report "
        "<published directory>",
        "```",
        "",
        "The manifest holds the revision, the case set, the document hashes, the "
        "prompts, the models, the settings, and the environment; each result "
        "beside it holds the index manifest and generation of every document, "
        "with the per-question rows the numbers came from. A run that measured "
        "the same things reproduces the manifest; a run that did not names the "
        "field that moved, and a changed model revision is reported on its own.",
        "",
    ]
    return "\n".join(lines)
