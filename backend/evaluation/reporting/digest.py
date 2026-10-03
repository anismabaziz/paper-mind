"""
One number a reviewer can act on: what a run cost in quality, seconds, and dollars.

Three questions are usually answered by three documents, and each of them can be
flattered by the other two being absent. A quality report leaves out the cases
that failed, a latency report leaves out the cases that were hard, and a cost
report leaves out everything the provider was never billed for. This module
answers them together, from the same cases, so a configuration cannot look cheap
by being asked fewer questions or look fast by failing the slow ones.

Two sources feed the digest and neither substitutes for the other. The
evaluation run says what the agreed case set produced, under versions it
records. The recorded traces say what readers actually waited for, in a shape
the answer path already redacts. A run with no trace to read says so in the cell
rather than leaving it empty, because an empty cell reads as a measurement that
found nothing.

The local half of the cost is stated in seconds and named as a bill of nobody's:
embedding and reranking run on the reader's own machine, so what they cost is
wall-clock time on one CPU, and a dollar figure for them would be a fiction. The
token counts and the provider's price are the only money in the digest, and both
are attributed to a named model.

A threshold is only ever crossed by a number that was measured, and a threshold
over a number the run did not produce is reported as unmeasurable rather than
skipped. A gate that quietly passes what it cannot read is a gate nobody is
watching.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from evaluation.answers import experiments
from evaluation.reporting import report, traces as trace_reader

DIGEST_NAME = "digest.md"
DIGEST_JSON_NAME = "digest.json"

#: The device the embedding and reranking models are loaded on. Nothing here can
#: measure this, because it is a property of the machine rather than of the run,
#: so it is declared here and printed as the assumption it is. It is CPU because
#: that is what the local services load onto; a machine with a GPU would be a
#: different assumption, and one worth stating out loud.
LOCAL_DEVICE = "cpu"

#: The answer metrics grouped the way a reader asks about them: what the answer
#: said, whether it can be checked, and whether the run knew to hold its tongue.
ANSWER_METRICS = ("correctness", "faithfulness")
CITATION_METRICS = ("citation_precision", "citation_recall")
ABSTENTION_METRIC = "abstention"

#: The waits a run and a trace file both report, so the two sit in one table
#: without being translated. The evaluation run keeps the cold start apart; a
#: trace file has no first case to single out, so its population is the whole
#: window and the run's cold start is never averaged into it.
MEASURED = ("retrieval_seconds", "first_token_seconds", "total_seconds")

#: What a digest says in place of a measurement it does not have. Carrying the
#: reason means a reader can tell a missing measurement from a zero.
UNMEASURED = "not measured by this run"

#: What the recorded half of the report says when there is no export file to
#: read. Spelled out rather than left blank, because a reader has to be able to
#: tell "no reader was traced" from "the readers who were traced were fast".
NO_TRACES = "no recorded traces were read for this report"

#: The two populations a run's latencies are split into, and which one describes
#: a reader who is not the first to arrive.
COLD_START = "cold_start"
STEADY_STATE = "steady_state"


def _results_by_id(results: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the results a comparison pairs up, by the configuration each measured."""
    return {result["experiment"]: result for result in results}


def _primary(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """
    Return the result a digest's own numbers describe.

    The shipped configuration, because that is what a reader is deciding about; a
    report that led with whichever variant happened to be measured last would make
    the baseline look like the experiment. A report with no baseline falls back to
    its first result rather than reporting nothing.
    """
    if not results:
        return {}
    return _results_by_id(results).get(experiments.BASELINE.id) or results[0]


def _metric(answers: dict[str, Any], name: str) -> dict[str, Any]:
    """Return one graded metric as it is reported, counts and mean together."""
    metric = answers.get(name) or {}
    return {
        "graded": metric.get("graded", 0),
        "scored": metric.get("scored", 0),
        "mean": metric.get("mean"),
        "passed": metric.get("passed", 0),
        "failed": metric.get("failed", 0),
        "unknown": metric.get("unknown", 0),
    }


def _failures(result: dict[str, Any]) -> dict[str, Any]:
    """
    Return what the run did not get right, by the outcome and the grader.

    Both halves are counted rather than dropped, because the number that decides
    whether a cheap configuration is worth having is what it gave up: a run that
    stopped answering hard questions improves every mean it reports and answers
    fewer people.
    """
    outcomes = {
        name: count for name, count in (result.get("outcomes") or {}).items() if count
    }
    graders: dict[str, int] = {}
    cases: list[str] = []
    for case in result.get("cases", []):
        rejected = report.failed_graders(case)
        if case.get("outcome") != "answered" or rejected:
            cases.append(case["id"])
        for name in rejected:
            graders[name] = graders.get(name, 0) + 1
    return {"outcomes": outcomes, "graders": graders, "cases": cases}


#: Each part of the quality table, and the evidence kind whose absence empties it.
QUALITY_EVIDENCE = {
    "answers": "answers",
    "citations": "citations",
    "abstention": "abstention",
}


def _unmeasured(manifest: dict[str, Any]) -> dict[str, str]:
    """
    Return each kind of answer evidence this run did not produce, and why not.

    A retrieval-only run measured no answers because nothing asked a model, and
    a table of empty cells would read as a run that asked one and got nothing.
    The reason travels with the report, so the digest quotes it rather than
    inventing a phrase of its own for a cell it knows nothing about.
    """
    stated = manifest.get("evidence") or {}
    missing = {}
    for kind, evidence in QUALITY_EVIDENCE.items():
        entry = stated.get(evidence) or {}
        if entry.get("measured"):
            continue
        # A manifest states what was not measured per experiment, so the reason
        # is the one that run gave rather than one invented here.
        reasons = [
            row.get("reason")
            for row in entry.get("not_measured", [])
            if row.get("reason")
        ]
        missing[kind] = reasons[0] if reasons else UNMEASURED
    return missing


def _quality(
    manifest: dict[str, Any], results: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Return the run's quality, broken down by the thing that decided each part."""
    if not results:
        return {
            "retrieval": None,
            "answers": {},
            "citations": {},
            "abstention": None,
            "failures": {"outcomes": {}, "graders": {}, "cases": []},
            "unmeasured": {kind: UNMEASURED for kind in QUALITY_EVIDENCE},
        }
    primary = _primary(results)
    retrieval = primary.get("retrieval") or {}
    answers = primary.get("answers") or {}
    unmeasured = _unmeasured(manifest)
    return {
        "unmeasured": unmeasured,
        "retrieval": {
            "questions": retrieval.get("questions", 0),
            "hit_rate": retrieval.get("hit_rate"),
            "recall": retrieval.get("recall"),
            "mrr": retrieval.get("mrr"),
            "ndcg": retrieval.get("ndcg"),
        },
        "answers": (
            {}
            if "answers" in unmeasured
            else {name: _metric(answers, name) for name in ANSWER_METRICS}
        ),
        "citations": (
            {}
            if "citations" in unmeasured
            else {name: _metric(answers, name) for name in CITATION_METRICS}
        ),
        "abstention": (
            None if "abstention" in unmeasured else _metric(answers, ABSTENTION_METRIC)
        ),
        "failures": _failures(primary),
    }


def _performance(
    result: dict[str, Any], trace_summary: dict[str, Any] | None
) -> dict[str, Any]:
    """
    Return how long a run and a reader each took, from the same three numbers.

    The run's cold start is reported beside its steady state rather than folded
    in: it is one case paying for a lazy load, and averaging it into the tail
    would report a slow service that no reader ever meets twice.
    """
    return {
        "evaluation": _run_performance(result),
        "traces": (
            trace_summary
            if trace_summary is not None
            else {"measured": False, "reason": NO_TRACES}
        ),
    }


def _run_performance(result: dict[str, Any]) -> dict[str, Any]:
    """
    Return one result's latencies, split into its cold start and its steady state.

    A retrieval-only report flattens its retrieval seconds to one summary with no
    phases, because it has one population and no model call to separate. That
    summary is read as the steady state, which is the population it is: every
    case of a retrieval run is a reader who had already paid for the index.
    """
    latency = result.get("latency") or {}
    measured = {
        phase: {name: (latency.get(phase) or {}).get(name, {}) for name in MEASURED}
        for phase in (COLD_START, STEADY_STATE)
    }
    if not any(measured[STEADY_STATE].values()) and latency.get("retrieval_seconds"):
        measured[STEADY_STATE]["retrieval_seconds"] = latency["retrieval_seconds"]
    return measured


def _local_cost(manifest: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """
    Return what embedding and reranking cost, in seconds on the reader's CPU.

    Nothing here is billed, so there is no price to report and the figure is
    declared as an assumption: the device, the two models, and the wall-clock
    seconds the run spent turning the sample Documents into vectors. The unit is
    seconds per Document because that is the unit a run measured and a stored
    report kept; a chunk count is not in the record, and a per-chunk figure
    derived from one would be a number nobody could reproduce.
    """
    index = {
        name: state for name, state in (result.get("index") or {}).items() if state
    }
    documents = len(index)
    seconds = round(
        sum(float(state.get("indexed_seconds") or 0.0) for state in index.values()), 6
    )
    reranker = manifest["models"].get("reranker") or {}
    return {
        "device": LOCAL_DEVICE,
        "billed_usd": 0.0,
        "embedding_model": manifest["models"]["embedding"]["id"],
        "rerank_model": reranker.get("id"),
        "rerank_applied": bool(manifest["settings"].get("rerank_enabled")),
        "documents": documents,
        "indexed_seconds": seconds,
        "seconds_per_document": round(seconds / documents, 6) if documents else None,
    }


def _cost(result: dict[str, Any], local: dict[str, Any]) -> dict[str, Any]:
    """Return the provider's side of the cost, beside what the reader's machine did."""
    cost = result.get("cost") or {}
    priced = int(cost.get("priced_cases") or 0)
    usd = float(cost.get("usd") or 0.0)
    return {
        "provider": {
            "input_tokens": cost.get("input_tokens", 0),
            "output_tokens": cost.get("output_tokens", 0),
            "usd": usd,
            "priced_cases": priced,
            "per_case_usd": round(usd / priced, 6) if priced else None,
            "model": cost.get("model"),
            "input_cost_per_million_usd": cost.get("input_cost_per_million_usd"),
            "output_cost_per_million_usd": cost.get("output_cost_per_million_usd"),
        },
        "traces": {},
        "local": local,
    }


def _provenance(
    manifest: dict[str, Any], results: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """
    Return the versions every number in the digest is only comparable under.

    The index generation is here because the same questions answered from a
    rebuilt generation are a different measurement, and nobody reading a quality
    number would think to ask which vectors produced it.
    """
    models = manifest["models"]
    judge = models.get("judge")
    primary = _primary(results)
    return {
        "dataset": manifest["dataset"]["version"],
        "split": manifest["dataset"]["split"],
        "questions": manifest["dataset"]["cases"],
        "revision": manifest["revision"]["revision"],
        "dirty": manifest["revision"].get("dirty", False),
        "subject": manifest["revision"].get("subject", ""),
        "citation_prompt": manifest["prompts"]["citation"],
        "judge_rubric": (judge or {}).get("rubric"),
        "generator": models.get("generator"),
        "judge": judge,
        "embedding": models["embedding"],
        "reranker": models.get("reranker"),
        "index": {
            name: state.get("index_generation")
            for name, state in (primary.get("index") or {}).items()
        },
        "configurations": sorted({result["experiment"] for result in results}),
    }


def digest(
    manifest: dict[str, Any],
    results: Sequence[dict[str, Any]],
    traces: Iterable[Path | str] | None = None,
    trace_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Return one run's quality, performance, and cost in a single record.

    ``traces`` names the recorded export files to read the other half from.
    Passing none is not an error: a run measured before the traces were being
    kept still has a half of a report, and says which half.

    ``trace_summary`` is the summary of an already-read population, for a caller
    that has one in hand. It wins over ``traces`` rather than adding to it,
    because two populations of readers added together would be a number nobody
    measured.
    """
    primary = _primary(results)
    if trace_summary is None:
        paths = [path for path in (traces or ()) if path is not None]
        trace_summary = trace_reader.summarize(paths) if paths else None
    cost = _cost(primary, _local_cost(manifest, primary))
    if trace_summary is not None:
        cost["traces"] = {
            **trace_summary["cost"],
            "input_tokens": trace_summary["tokens"]["input"],
            "output_tokens": trace_summary["tokens"]["output"],
            "traces": trace_summary["traces"],
        }
    return {
        "provenance": _provenance(manifest, results),
        "quality": _quality(manifest, results),
        "performance": _performance(primary, trace_summary),
        "cost": cost,
        "configurations": _configurations(results),
    }


def _configurations(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Return what each configuration changed and what it cost, for a report with several.

    A run that measured one configuration has nothing to compare it against, so
    it gets an empty table rather than a column of zeroes: a row that says
    nothing moved against nothing is noise a reader has to learn to skip.
    """
    if len(results) < 2:
        return []
    baseline = next(
        (
            result
            for result in results
            if result["experiment"] == experiments.BASELINE.id
        ),
        results[0],
    )
    rows = compare([baseline] * len(results), results)
    for row in rows:
        row["reference"] = row["experiment"] == baseline["experiment"]
    return rows


def _duration(seconds: float) -> str:
    """
    Return a window of time in the unit a reader would say it in.

    Traces are written whenever an application runs, so the window between the
    first and the last of them is however long the machine was up. In seconds
    that is a number nobody reads and compares against nothing.
    """
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} minutes"
    return f"{seconds / 3600:.1f} hours"


def _cell(summary: Any) -> str:
    """Return a percentile summary as the p50/p95 pair a table cell wants."""
    if not isinstance(summary, dict) or summary.get("p50") is None:
        return "no samples"
    return f"{summary['p50']:.2f} / {summary['p95']:.2f}"


def _quality_table(measured: dict[str, Any]) -> list[str]:
    """Return the quality half, with every case the run could not decide beside it."""
    quality = measured["quality"]
    lines = ["## Quality", ""]
    unmeasured = quality.get("unmeasured") or {}
    for kind, reason in sorted(unmeasured.items()):
        lines.append(f"- **{kind}**: not measured — {reason}")
    rows = [
        (group, name, quality[group][name])
        for group in ("answers", "citations")
        for name in sorted(quality[group])
    ]
    if not unmeasured.get("abstention") and quality["abstention"]:
        rows.append(("abstention", "abstention", quality["abstention"]))
    if not rows:
        if unmeasured:
            lines.append("")
        lines.append(
            "This run measured no answers, so it has no answer, citation, or "
            "abstention quality to report."
        )
        retrieval = quality["retrieval"]
        if retrieval:
            lines.append("")
            lines.append(
                f"Retrieval over {retrieval['questions']} questions: hit@k "
                f"{report.number_cell(retrieval['hit_rate'])}, recall "
                f"{report.number_cell(retrieval['recall'])}, MRR {report.number_cell(retrieval['mrr'])}, "
                f"nDCG {report.number_cell(retrieval['ndcg'])}."
            )
        return lines + [""] + _failures_lines(quality["failures"])
    lines += [
        "",
        "| Kind | Measure | Mean | Scored | Graded | Unknown | Failed |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for group, name, metric in rows:
        lines.append(
            f"| {group} | {name.replace('_', ' ')} | {report.number_cell(metric.get('mean'))} | "
            f"{metric.get('scored', 0)} | {metric.get('graded', 0)} | "
            f"{metric.get('unknown', 0)} | {metric.get('failed', 0)} |"
        )
    retrieval = quality["retrieval"]
    if retrieval:
        lines += [
            "",
            f"Retrieval over {retrieval['questions']} questions: hit@k "
            f"{report.number_cell(retrieval['hit_rate'])}, recall {report.number_cell(retrieval['recall'])}, "
            f"MRR {report.number_cell(retrieval['mrr'])}, nDCG {report.number_cell(retrieval['ndcg'])}.",
        ]
    return lines + [""] + _failures_lines(quality["failures"])


def _failures_lines(failures: dict[str, Any]) -> list[str]:
    """
    Return what the run did not get right, in prose rather than in a table.

    A retrieval run failed nothing, because it answered nothing, and a count of
    zero in a failure column would read as a result rather than as an absence.
    """
    outcomes = ", ".join(
        f"{name} {count}"
        for name, count in failures["outcomes"].items()
        if name != "answered"
    )
    graders = ", ".join(
        f"{name} {count}" for name, count in failures["graders"].items()
    )
    answered = failures["outcomes"].get("answered")
    return [
        f"{f'{answered} cases were answered. ' if answered else ''}"
        f"What did not go right: {outcomes or 'nothing'}; "
        f"graders that rejected an answer: {graders or 'none'}. "
        f"The {len(failures['cases'])} cases behind those numbers are listed in the "
        "report beside this one."
    ]


def _performance_table(measured: dict[str, Any]) -> list[str]:
    """Return the waits, from the run and from the recorded requests beside them."""
    performance = measured["performance"]
    lines = [
        "## Performance",
        "",
        "Seconds, as p50 / p95. A p50 alone is the answer most readers get and a "
        "p95 alone is the one they complain about.",
        "",
        "| Where | Retrieval | First token | Total |",
        "| --- | --- | --- | --- |",
    ]
    evaluation = performance["evaluation"]
    for phase, label in (
        (COLD_START, "run, cold start"),
        (STEADY_STATE, "run, steady"),
    ):
        lines.append(
            f"| {label} | {_cell(evaluation[phase]['retrieval_seconds'])} | "
            f"{_cell(evaluation[phase]['first_token_seconds'])} | "
            f"{_cell(evaluation[phase]['total_seconds'])} |"
        )
    recorded = performance["traces"]
    if not recorded.get("measured", True):
        lines.append(f"| recorded requests | - | - | {recorded['reason']} |")
    else:
        latency = recorded["latency"]
        lines.append(
            f"| recorded requests | {_cell(latency['retrieval_seconds'])} | "
            f"{_cell(latency['first_token_seconds'])} | "
            f"{_cell(latency['total_seconds'])} |"
        )
        rate = recorded["throughput"]["traces_per_minute"]
        outcomes = ", ".join(
            f"{name} {count}" for name, count in sorted(recorded["outcomes"].items())
        )
        lines += [
            "",
            f"What those {recorded['traces']} requests ended as: {outcomes}.",
            f"They arrived over "
            f"{_duration(recorded['throughput']['window_seconds'])}, which is "
            + (f"{rate:.2f} a minute." if rate is not None else "no measurable rate."),
        ]
    return lines


def _configuration_table(measured: dict[str, Any]) -> list[str]:
    """Return every configuration beside what it gained and what it cost."""
    lines = [
        "## Configurations",
        "",
        "Change against the shipped configuration, which is compared with itself so "
        "the other rows read as differences rather than as absolutes.",
        "",
        "| Configuration | nDCG | Correctness | Retrieval p50 (s) | Total p50 (s) | "
        "Total p95 (s) | Cost ($) | Cost a case ($) | Indexing (s) | What did not go right |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in measured["configurations"]:
        failed = (
            ", ".join(
                f"{name} {count}" for name, count in row["failures"]["after"].items()
            )
            or "nothing"
        )
        name = row["experiment"]
        lines.append(
            f"| `{name}{' (reference)' if row.get('reference') else ''}` | "
            f"{_signed(row['quality']['ndcg'])} | "
            f"{_signed(row['quality']['correctness'])} | "
            f"{_signed(row['latency']['retrieval_seconds_p50'])} | "
            f"{_signed(row['latency']['total_seconds_p50'])} | "
            f"{_signed(row['latency']['total_seconds_p95'])} | "
            f"{_signed(row['cost']['usd'], 4)} | "
            f"{_signed(row['cost']['per_case_usd'], 4)} | "
            f"{_signed(row['cost']['indexed_seconds'])} | {failed} |"
        )
    return lines


def _signed(value: Any, places: int = 3) -> str:
    """Return a change as a signed number, so a reader sees which way it moved."""
    if value is None:
        return "-"
    return f"{value:+.{places}f}"


def _cost_table(measured: dict[str, Any]) -> list[str]:
    """Return the money, and the seconds that cost nobody anything but CPU."""
    cost = measured["cost"]
    provider = cost["provider"]
    local = cost["local"]
    if not provider["priced_cases"]:
        return [
            "## Cost",
            "",
            "No model was called, so this run was billed nothing. The provider, the "
            "tokens, and the dollars are absent from this report rather than "
            "reported as zero, because a run that asked nobody is not a run that "
            "asked cheaply.",
            "",
            _local_line(local),
        ]
    lines = [
        "## Cost",
        "",
        f"Provider: {provider['input_tokens']} input and "
        f"{provider['output_tokens']} output tokens over "
        f"{provider['priced_cases']} answered cases at `{provider['model']}`, "
        f"${report.number_cell(provider['usd'], 4)} in total and "
        f"${report.number_cell(provider['per_case_usd'], 4)} per answered case.",
    ]
    if cost["traces"].get("traces"):
        recorded = cost["traces"]
        lines.append(
            f"Recorded requests outside the case set: "
            f"{recorded['input_tokens']} input and "
            f"{recorded['output_tokens']} output tokens over "
            f"{recorded['priced_calls']} "
            f"calls at `{recorded['model']}`, "
            f"${report.number_cell(recorded['usd'], 4)} in total and "
            f"${report.number_cell(recorded['per_trace_usd'], 4)} a call."
        )
    lines += ["", _local_line(local)]
    return lines


def _local_line(local: dict[str, Any]) -> str:
    """Return the local half of the cost: seconds on a CPU, and nobody's bill."""
    return (
        f"Local compute is declared, not billed: embedding "
        f"`{local['embedding_model']}` and reranking with "
        f"`{local['rerank_model']}` "
        + ("were applied" if local["rerank_applied"] else "were not applied")
        + f", on {local['device']}, took {local['indexed_seconds']:.1f}s to index "
        f"{local['documents']} documents"
        + (
            f" ({local['seconds_per_document']:.1f}s a document)."
            if local["seconds_per_document"] is not None
            else "."
        )
    )


def render(measured: dict[str, Any]) -> str:
    """Return the digest as the markdown a reviewer reads first."""
    provenance = measured["provenance"]
    generator = provenance["generator"] or {}
    judge = provenance["judge"] or {}
    generating = (
        f"Generating model `{generator['id']}` on {generator['provider']}"
        if generator
        else "No model generated."
    )
    lines = [
        f"# Quality, latency, and cost: case set {provenance['dataset']}",
        "",
        f"Split `{provenance['split']}`, {provenance['questions']} questions, on revision "
        f"`{(provenance['revision'] or 'unknown')[:12]}`"
        + (", with uncommitted changes" if provenance["dirty"] else "")
        + f" — {provenance['subject']}",
        generating
        + (
            f", judged by `{judge.get('id')}` on {judge.get('provider')} "
            f"(rubric {judge.get('rubric')})"
            if judge
            else (", not judged" if generator else "")
        ),
        f"Citation prompt `{provenance['citation_prompt']}`, embedding "
        f"`{provenance['embedding']['id']}`"
        + (
            f" at `{provenance['embedding']['revision']}`"
            if provenance["embedding"].get("revision")
            else ""
        ),
        "",
        "These three answers come from the same cases. A configuration cannot look "
        "cheap here by being asked fewer questions, and cannot look fast by "
        "failing the slow ones: the failures are in the quality table and the "
        "abstentions are counted as results.",
        "",
    ]
    lines += _quality_table(measured) + [""] + _performance_table(measured)
    if measured["configurations"]:
        lines += [""] + _configuration_table(measured)
    lines += [""] + _cost_table(measured)
    lines += [
        "",
        "## Reproducing this digest",
        "",
        "```",
        *_reproduction_lines(measured),
        "```",
        "",
    ]
    return "\n".join(lines)


def _reproduction_lines(measured: dict[str, Any]) -> list[str]:
    """
    Return the commands that reproduce this digest, in the order they are run.

    Two commands, because they answer two questions. The first measures the run
    again, so it is the run's own command: an answer-path run names the models
    that wrote and judged it, and a retrieval-only run names ``--ablate`` and no
    key, because nothing asked a model. The second re-derives the digest from a
    report already on disk, which is what a reviewer runs.
    """
    provenance = measured["provenance"]
    generator = provenance["generator"] or {}
    judge = provenance["judge"] or {}
    if generator:
        return [
            "docker compose up db qdrant",
            "export PAPERMIND_EVAL_GENERATOR_API_KEY=...",
            *(["export PAPERMIND_EVAL_JUDGE_API_KEY=..."] if judge else []),
            "uv run python -m evaluation.cli --live \\",
            f"  --provider {generator.get('provider', '')} "
            f"--model {generator.get('id', '')} \\",
            *(
                [
                    f"  --judge-provider {judge.get('provider', '')} "
                    f"--judge-model {judge.get('id', '')} \\"
                ]
                if judge
                else []
            ),
            "  --report <directory>",
            "uv run python -m evaluation.cli --render <directory>",
        ]
    return [
        "docker compose up qdrant",
        "uv run python -m evaluation.cli --live --ablate --report <directory>",
        "uv run python -m evaluation.cli --render <directory>",
    ]


def write(directory: Path | str, measured: dict[str, Any]) -> str:
    """
    Write the digest beside the report it describes, as markdown and as JSON.

    Both forms, because the markdown is what a person reads and the JSON is what
    a threshold gate reads; a gate parsing the prose would break the day a word
    in a sentence changed.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rendered = render(measured)
    (directory / DIGEST_NAME).write_text(rendered, encoding="utf-8")
    (directory / DIGEST_JSON_NAME).write_text(
        json.dumps(measured, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rendered


def _delta(before: Any, after: Any) -> float | None:
    """Return how far a number moved, or None when either side never measured it."""
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
        return None
    return after - before


def _mean(answers: dict[str, Any], name: str) -> Any:
    """Return one answer metric's mean, or None when the run did not grade it."""
    return ((answers.get(name) or {}).get("mean")) if answers else None


def _steady(result: dict[str, Any], name: str) -> Any:
    """
    Return one steady-state measurement of a result, wherever that run kept it.

    An answer-path run splits its timings into a cold start and a steady state;
    a retrieval-only run has one population and flattens its retrieval seconds to
    the top level. Both are compared on the steady state, because that is the
    population that describes a reader who is not the first to arrive.
    """
    latency = result.get("latency") or {}
    summary = (latency.get(STEADY_STATE) or {}).get(name) or {}
    if not summary and name == "retrieval_seconds":
        summary = latency.get("retrieval_seconds") or {}
    return summary


def _row(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """
    Return one configuration's quality change beside what it cost to get it.

    The failures of both sides travel in the row, because a variant that answers
    fewer questions improves every mean in it, and a table that showed only the
    improved means would be the reason that got shipped.
    """
    return {
        "experiment": after["experiment"],
        "family": after.get("family", ""),
        "quality": {
            "hit_rate": _delta(
                (before.get("retrieval") or {}).get("hit_rate"),
                (after.get("retrieval") or {}).get("hit_rate"),
            ),
            "ndcg": _delta(
                (before.get("retrieval") or {}).get("ndcg"),
                (after.get("retrieval") or {}).get("ndcg"),
            ),
            "correctness": _delta(
                _mean(before.get("answers") or {}, "correctness"),
                _mean(after.get("answers") or {}, "correctness"),
            ),
            "faithfulness": _delta(
                _mean(before.get("answers") or {}, "faithfulness"),
                _mean(after.get("answers") or {}, "faithfulness"),
            ),
        },
        "latency": {
            f"{name}_{quantile}": _delta(
                _steady(before, name).get(quantile),
                _steady(after, name).get(quantile),
            )
            for name in MEASURED
            for quantile in ("p50", "p95")
        },
        "cost": {
            "usd": _delta(
                (before.get("cost") or {}).get("usd"),
                (after.get("cost") or {}).get("usd"),
            ),
            "input_tokens": _delta(
                (before.get("cost") or {}).get("input_tokens"),
                (after.get("cost") or {}).get("input_tokens"),
            ),
            "output_tokens": _delta(
                (before.get("cost") or {}).get("output_tokens"),
                (after.get("cost") or {}).get("output_tokens"),
            ),
            "per_case_usd": _delta(_per_case(before), _per_case(after)),
            "indexed_seconds": _delta(
                _indexed_seconds(before), _indexed_seconds(after)
            ),
        },
        "failures": {
            "before": _failures(before)["outcomes"],
            "after": _failures(after)["outcomes"],
        },
        "cases": {
            "before": len(before.get("cases", [])),
            "after": len(after.get("cases", [])),
        },
    }


def _indexed_seconds(result: dict[str, Any]) -> float:
    """
    Return what a configuration spent turning the documents into vectors.

    The only cost a retrieval-only configuration has, and a real one: a chunking
    or embedding variant has to index every document again, so the seconds are
    the price of the variant rather than a footnote about the machine.
    """
    return round(
        sum(
            float(state.get("indexed_seconds") or 0.0)
            for state in (result.get("index") or {}).values()
            if state
        ),
        6,
    )


def _per_case(result: dict[str, Any]) -> float | None:
    """Return what one answered case cost, or None when none was priced."""
    cost = result.get("cost") or {}
    priced = int(cost.get("priced_cases") or 0)
    return round(float(cost.get("usd") or 0.0) / priced, 6) if priced else None


def compare(
    before: Sequence[dict[str, Any]], after: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Return what each configuration changed and what it cost to change it.

    A configuration is measured against itself where both sides have it, and
    against the shipped configuration where only the second side does, so a
    report that re-ran one variant beside a full set of results still gets a row
    for it. The shipped configuration ends up compared with itself, which is
    what makes the other rows' differences readable as differences rather than as
    absolutes. A configuration that was asked different questions on the two
    sides gets its row anyway with its case counts, because a comparison over a
    different case set is a thing a reader has to be able to see.
    """
    earlier, later = _results_by_id(before), _results_by_id(after)
    reference_id = experiments.BASELINE.id
    rows = []
    for name, result in later.items():
        reference = earlier.get(name) or earlier.get(reference_id)
        if reference is None:
            continue
        rows.append(_row(reference, result))
    return rows


def _lookup(measured: dict[str, Any], path: str) -> Any:
    """Return the value at a dotted path in the digest, or None when it is not there."""
    value: Any = measured
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def regressions(
    measured: dict[str, Any], thresholds: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Return the thresholds this digest crossed, and the ones it could not measure.

    A bound is either a floor or a ceiling and the name says which, so a
    threshold file cannot claim that quality may fall to zero because somebody
    wrote ``max`` where they meant ``min``.

    Every path is a path into the digest, written out in full: the steady state
    is named rather than assumed, so a bound is legible on its own and a typo
    is a path that resolves to nothing. A bound over a number this run did not
    produce is reported with that reason rather than skipped, because a gate that
    passes what it cannot read is not a gate.
    """
    crossed = []
    for path, bound in sorted(thresholds.items()):
        value = _lookup(measured, path)
        if value is None or not isinstance(value, (int, float)):
            crossed.append(
                {
                    "threshold": path,
                    "reason": UNMEASURED,
                    "measured": None,
                    "bound": bound,
                }
            )
            continue
        for direction in ("min", "max"):
            if direction not in bound:
                continue
            limit = bound[direction]
            missed = value < limit if direction == "min" else value > limit
            if missed:
                crossed.append(
                    {
                        "threshold": path,
                        "measured": value,
                        "bound": limit,
                        "direction": direction,
                        "reason": f"below the declared {direction} of {limit}"
                        if direction == "min"
                        else f"above the declared {direction} of {limit}",
                    }
                )
    return crossed
