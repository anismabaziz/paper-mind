"""
Opt-in live evaluation CLI.

Without ``--live`` this refuses to run: the point of the evaluator is a
measured number, but a live run indexes real documents and answers them with a
paid provider, so it only ever happens on purpose.

Keys are read from the environment, never from the command line, because a
command line is visible to every process on the machine and lands in shell
history. The generator and the judge are configured separately — different
providers, different models, different keys — so a run can generate with one
account and grade with another, and a retrieval-only run never has to hold a
key it does not need.

What a run reports is split so a failure cannot flatter a number. Retrieval
reports hit rate, recall, MRR, and nDCG per question and in aggregate. Answers
report what the deterministic graders decided, what the judge decided, and how
many cases each of those could not decide at all. Latency reports the run's
first case apart from the rest, because that is the one paying for a lazy load.
Cost is reported per answered case from the tokens the run measured.

Free local (no keys, default):
    docker compose up qdrant                          # http://localhost:6333
    uv run python -m evaluation.cli --live --no-judge  # retrieval only, no LLM
    uv run python -m evaluation.cli --live --ablate --report evaluation/reports/base

With generation and a model judge:
    export PAPERMIND_EVAL_GENERATOR_API_KEY=...
    export PAPERMIND_EVAL_JUDGE_API_KEY=...
    uv run python -m evaluation.cli --live

``--report`` writes what a reviewer reads without running anything: a manifest
holding the revision, the case set, the document hashes, the index manifests,
the prompts, the models, the settings, and the environment, plus one file per
experiment and a summary. ``--ablate`` measures the declared retrieval
experiments against the same labeled cases instead of asking a model, so it
needs no key; ``--compare-report`` says whether this run reproduces a
published one and names the field that moved.

``--render`` re-reads a report that is already on disk and writes the digest
beside it: quality, latency, and cost in one document, with the recorded traces
of real requests as its other half. It asks no model, holds no key, and starts
no index, so a reviewer or a CI job can re-derive the digest from the committed
report alone. ``--traces`` names the export files to read; whatever they
summarize is stored in the report directory, because the raw traces are an
operator's file and the summary is the report's. ``--check`` compares the digest
against bounds somebody wrote down and exits non-zero when one is crossed.

Qdrant local URL is ``http://localhost:6333`` on the host
(``http://qdrant:6333`` inside compose, via ``QDRANT_URL``).
"""

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path


import settings as settings_module
from services.accounts.chat_settings_service import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    SettingsError,
    model_for,
    validate,
)
from services.llm.base import ChatCredentials, is_context_fallback
from services.llm.factory import build_chat_provider
from settings import Settings

from evaluation.answers import ablations, experiments
from evaluation.answers.dataset import REPORTED, SPLITS, load_dataset
from evaluation.answers.evaluator import DEFAULT_K, evaluate
from evaluation.answers.harness import build_environment, remove_documents
from evaluation.answers.judge import Judge, JudgeSettings
from evaluation.reporting import digest, report, traces

EVAL_PREFIX = "eval-"

#: Environment variables the run reads its keys from. Neither is ever accepted
#: as an argument, and neither is printed: the run record names the provider
#: and the model, which is what makes a result comparable.
GENERATOR_API_KEY_ENV = "PAPERMIND_EVAL_GENERATOR_API_KEY"
JUDGE_API_KEY_ENV = "PAPERMIND_EVAL_JUDGE_API_KEY"

#: What a judge call that failed comes back as: a reply that is not a verdict,
#: so the case is reported Unknown rather than graded or crashed on.
JUDGE_FAILED = "the judge's call failed"

#: Seconds to wait before each provider call. A rate-limited account measures
#: its budget in tokens per day, and a run that asks for everything at once
#: spends it and then reports the cases it could not afford as provider
#: failures — a number about the account, not about the application. The wait
#: is the harness's own pacing; nothing in the answer path knows about it.
DEFAULT_PACE_SECONDS = 6.0

#: The pause this process is running with, which ``--pace`` sets.
PROVIDER_PACE_SECONDS = DEFAULT_PACE_SECONDS

#: What a render stores beside the report: the summary of the recorded requests
#: the digest's runtime half was read from. Named separately from the digest's
#: own files because it is the input, and re-rendering reads it back.
TRACE_SUMMARY_NAME = "traces-summary.json"

#: How many output tokens the judge may use. A reasoning model spends its
#: budget thinking through the rubric before it answers, and an empty reply is
#: what a budget that ran out looks like. The verdict the judge is asked for is
#: one word, but the answer has to fit beside the thinking that produced it, so
#: the judge runs under a bigger budget than the app's answers do.
JUDGE_MAX_OUTPUT_TOKENS = 4096

#: How long to wait before asking the judge again, and how many times to ask.
#: A judge is three calls per case on a rate-limited account, and a throttle is
#: the one failure a pause fixes; a rejected key or a malformed prompt is not,
#: so only a transient failure spends another call.
JUDGE_BACKOFF_SECONDS = (2.0, 6.0, 15.0)

#: The judge's one-shot instruction, carried as the question so the provider's
#: own prompt framing stays the one the app uses.
JUDGE_INSTRUCTION = (
    "You are a strict evaluation judge. Follow the output format exactly. "
    "Reply with the single word the prompt asks for and nothing else."
)


def secret_from_env(name: str) -> str:
    """Return a key from the environment, or the empty string when unset."""
    return (os.environ.get(name) or "").strip()


def require_key(name: str) -> str:
    """Return a key from the environment, or exit naming the variable to set."""
    value = secret_from_env(name)
    if not value:
        sys.exit(f"{name} is not set. Export the key for the provider you chose.")
    return value


def pace() -> None:
    """Wait the run's own pause between provider calls."""
    if PROVIDER_PACE_SECONDS > 0:
        time.sleep(PROVIDER_PACE_SECONDS)


def generator_factory():
    """Return the factory the run's generator is built through."""

    def factory(credentials: ChatCredentials):
        """Build the generator for the credentials the answer path supplies."""
        pace()
        return build_chat_provider(credentials, use_cache=False)

    return factory


def _judge_provider(provider: str, model: str, api_key: str):
    """Return the provider the judge answers through, under its own credentials."""
    model_definition = model_for(provider, model)
    return build_chat_provider(
        ChatCredentials(
            provider=provider,
            model=model,
            api_key=api_key,
            verification_timeout_seconds=model_definition.timeout_seconds,
            budget=replace(
                model_definition.chat_budget(),
                max_output_tokens=JUDGE_MAX_OUTPUT_TOKENS,
            ),
        ),
        use_cache=False,
    )


def build_evaluation_judge(provider: str, model: str, api_key: str) -> Judge:
    """
    Return the judge a run grades with, and the identity the run records.

    The judge is built through the app's own provider factory with its own
    credentials, so a run can generate with one account and grade with another,
    and a judge that is not Google reaches the provider it was asked for. A call
    that failed comes back as the provider's refusal to answer, which is
    reported as a failed judge rather than graded as a verdict.
    """
    return Judge(
        settings=JudgeSettings(provider=provider, model=model),
        grade=_judge_callable(provider, model, api_key),
    )


def _judge_callable(provider: str, model: str, api_key: str) -> Callable[[str], str]:
    """
    Return the callable that sends one rubric prompt to the judge's model.

    A call that fails — a throttle, a timeout, a provider that fell back to
    quoting the request back — comes back as a reply that is not a verdict
    rather than as an exception. A throttled judge is a case nothing could be
    decided about, which the report already has a word for: Unknown. Raising
    instead would throw away a half-finished run over one call.

    A call that produced no verdict is asked again after a pause, because a
    rate limit is the one failure that waiting fixes, and a judged run that
    quietly decided a third of its cases is a worse report than a slower one.
    Only a bounded number of extra calls: an account that is out of credit
    should end the run's judging, not spend it.
    """
    judge_provider = _judge_provider(provider, model, api_key)

    def grade(prompt: str) -> str:
        """Return the judge's reply, or a non-verdict when the call failed."""
        pace()
        for attempt in range(len(JUDGE_BACKOFF_SECONDS) + 1):
            if attempt:
                time.sleep(JUDGE_BACKOFF_SECONDS[attempt - 1])
            try:
                reply = judge_provider.generate_response(JUDGE_INSTRUCTION, prompt)
            except Exception as exc:  # noqa: BLE001 - any failure is an Unknown
                if not is_transient_error(exc):
                    return JUDGE_FAILED
                continue
            if reply in (judge_provider.FALLBACK_ANSWER,) or is_context_fallback(reply):
                # A provider that failed hands back the context it was given
                # rather than raising, and a throttle looks like every other
                # failure from here. The call is asked again; the reply that
                # stands in for a judge that never answered carries neither the
                # failure's own words nor the key it tried to echo back.
                continue
            return reply
        return JUDGE_FAILED

    return grade


def with_rerank(base: Settings, enabled: bool) -> Settings:
    """
    Return the configuration with the reranker gate set for this run.

    The gate is configuration, not a branch in the answer path: an ablation
    changes the setting the app reads, so the run still travels the same code a
    reader's app would.
    """
    return base.model_copy(
        update={"rerank": base.rerank.model_copy(update={"enabled": enabled})}
    )


def _environment(
    dataset,
    app_settings: Settings,
    documents_prefix: str,
    rerank_enabled: bool | None = None,
):
    """
    Build the application a run measures: real documents, real retrieval.

    The reranker follows the settings' gate, because that gate is what a reader
    runs with. An ablation passes ``rerank_enabled=True`` to load the reranker
    even when the gate is off, so that a variant can turn it on for one call
    over an index the other variants share.
    """
    from composition import Services
    from services.retrieval.reranker import RerankerService
    from services.retrieval.vector_service import VectorService

    settings_module.set_settings(app_settings)
    services = Services.from_settings(app_settings)
    vector_service = services.vector_service
    if rerank_enabled is not None:
        vector_service = VectorService(
            services.vector_service.store,
            RerankerService(
                app_settings.rerank.rerank_model,
                revision=app_settings.rerank.revision,
                trust_remote_code=app_settings.rerank.trust_remote_code,
                enabled=rerank_enabled,
            ),
            embedding_service=services.embedding_service,
        )
    return build_environment(
        dataset,
        settings=app_settings,
        session_factory=None,
        storage=services.storage,
        embedding_service=services.embedding_service,
        vector_service=vector_service,
        chat_provider_factory=generator_factory(),
        documents_prefix=documents_prefix,
    )


def _run_once(
    dataset,
    app_settings: Settings,
    *,
    provider: str,
    model: str,
    generator_key: str,
    judge: Judge | None = None,
    k: int = DEFAULT_K,
    calibrate: bool = True,
    split: str = REPORTED,
    include_faults: bool = False,
) -> dict:
    """Index the documents, run one split of the set, and clean up after itself."""
    environment = _environment(dataset, app_settings, EVAL_PREFIX)
    try:
        started = time.monotonic()
        # The generator is built inside the run, from the model and the key,
        # through the app's own provider factory: the run asks the same factory
        # a reader's request does rather than building a provider of its own.
        report = evaluate(
            dataset,
            environment,
            model=model_for(provider, model),
            api_key=generator_key,
            judge=judge,
            k=k,
            calibrate=calibrate,
            split=split,
            include_faults=include_faults,
        )
        report.run["seconds"] = time.monotonic() - started
        return report.as_dict()
    finally:
        remove_documents(environment)
        settings_module.set_settings(None)


def run(
    live: bool,
    use_judge: bool,
    k: int,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    judge_provider: str = DEFAULT_PROVIDER,
    judge_model: str = DEFAULT_MODEL,
    rerank: bool | None = None,
    compare_rerank: bool = False,
    calibrate: bool = True,
    split: str = REPORTED,
    include_faults: bool = False,
) -> dict:
    """
    Run the labeled case set through the production answer path.

    ``provider``/``model`` are the generator's and ``judge_provider``/``judge_model``
    the judge's. They are separate settings because a run may generate with one
    account and grade with another, and because grading with a different model
    is a measurement choice rather than an accident.
    """
    if not live:
        sys.exit(
            "Refusing to run a live evaluation by default. Add --live to "
            "embed, retrieve, generate, and judge against the real providers."
        )
    validate(provider, model)
    if use_judge:
        validate(judge_provider, judge_model)
    settings_module.validate()
    generator_key = require_key(GENERATOR_API_KEY_ENV)
    grader = (
        build_evaluation_judge(
            judge_provider, judge_model, require_key(JUDGE_API_KEY_ENV)
        )
        if use_judge
        else None
    )

    dataset = load_dataset()
    base_settings = settings_module.get_settings()
    if compare_rerank:
        return {
            "rerank": {
                label: _run_once(
                    dataset,
                    with_rerank(base_settings, enabled),
                    provider=provider,
                    model=model,
                    generator_key=generator_key,
                    judge=grader,
                    k=k,
                    calibrate=calibrate,
                    split=split,
                    include_faults=include_faults,
                )
                for label, enabled in (("off", False), ("on", True))
            }
        }
    return _run_once(
        dataset,
        base_settings if rerank is None else with_rerank(base_settings, rerank),
        provider=provider,
        model=model,
        generator_key=generator_key,
        judge=grader,
        k=k,
        calibrate=calibrate,
        split=split,
        include_faults=include_faults,
    )


def run_ablations(
    dataset,
    live: bool,
    k: int = DEFAULT_K,
    split: str = REPORTED,
) -> list[dict]:
    """
    Measure the declared experiments against the same labeled cases.

    Retrieval only, so this measures nothing a model wrote and says so: the
    report it produces carries no answer, citation, abstention, token, or dollar
    figure. What it does carry is a column per experiment, a per-question row
    for every number, and a comparison against the baseline that names the
    questions that moved.
    """
    if not live:
        sys.exit(
            "Refusing to run a live evaluation by default. Add --live to index "
            "the documents and retrieve the questions for real."
        )
    settings_module.validate()
    base_settings = settings_module.get_settings()
    cases = dataset.cases_for(split)
    measured = ablations.run(
        experiments.REGISTRY,
        cases,
        lambda experiment: _environment(
            dataset,
            experiment.settings(base_settings),
            f"{EVAL_PREFIX}{experiment.id}-",
            rerank_enabled=True,
        ),
        dataset=dataset,
        k=k,
        close=_release,
    )
    results = []
    for baseline, variant in zip(measured, measured[1:]):
        results.append(
            {
                **variant.as_dict(),
                "comparison": ablations.compare(baseline, variant).as_dict(),
            }
        )
    # The baseline is compared with itself: nothing moved, and saying so is
    # what makes the other columns' differences readable as differences.
    results.insert(
        0,
        {
            **measured[0].as_dict(),
            "comparison": ablations.compare(measured[0], measured[0]).as_dict(),
        },
    )
    return results


def _release(environment) -> None:
    """Delete an environment's Documents and their vectors, restoring the app."""
    remove_documents(environment)
    settings_module.set_settings(None)


def result_of_run(run_record: dict) -> dict:
    """
    Return one answer-path run in the shape a report stores results in.

    The evidence section is what a reader goes to first: it says which of the
    numbers in the result a model decided, which a deterministic grader decided,
    and which the run could not measure at all. The latency is flattened to the
    retrieval measurement a table cell wants, because an answer-path run's
    steady state is the only population of it that describes a reader waiting.
    """
    latency = dict(run_record["latency"])
    latency["retrieval_seconds"] = latency["steady_state"]["retrieval_seconds"]
    return {
        "experiment": experiments.BASELINE.id,
        "family": experiments.BASELINE.family,
        "dataset": run_record["run"]["dataset"],
        "split": run_record["run"]["split"],
        "held_back": run_record["run"]["held_back"],
        "index": run_record["run"]["documents"],
        "retrieval": run_record["retrieval"],
        "answers": run_record["answers"],
        "outcomes": run_record["outcomes"],
        "latency": latency,
        "cost": run_record["cost"],
        "calibration": run_record.get("calibration"),
        "judge": run_record["run"].get("judge"),
        "cases": run_record["cases"],
        "evidence": report.evidence_from_run(run_record),
    }


def publish(
    dataset,
    results: list[dict],
    directory,
    run_records: list[dict] | None = None,
) -> Path:
    """Write a report directory a reviewer can read without running anything."""
    settings = settings_module.get_settings()
    manifest = report.build_manifest(
        dataset=dataset,
        settings=settings,
        results=results,
        run_records=run_records or [],
        revision=report.git_revision(),
    )
    return report.write_report(Path(directory), manifest, results)


def _seconds(summary: dict) -> str:
    """Return a latency summary as p50/p95 in seconds, or what it has instead."""
    if summary["p50"] is None:
        return "no samples"
    return f"p50 {summary['p50']:.2f}s p95 {summary['p95']:.2f}s"


def _mean(metric: dict) -> str:
    """Return one answer metric as its mean and how many cases earned it."""
    if metric["mean"] is None:
        return (
            f"nothing scored ({metric['graded']} graded, {metric['unknown']} unknown)"
        )
    return (
        f"{metric['mean']:.2f} mean over {metric['scored']}/{metric['graded']} "
        f"({metric['passed']} full marks)"
    )


def _print(run_report: dict) -> None:
    """Print a run as a short operator-facing summary."""
    if "rerank" in run_report:
        for label, run_record in run_report["rerank"].items():
            print(f"--- rerank {label} ---")
            _print(run_record)
        return
    run_record = run_report["run"]
    retrieval = run_report["retrieval"]
    judged_by = run_record.get("judge")
    held_back = run_record["held_back"]
    print(
        f"Run: {run_record['provider']}/{run_record['model']}, prompt "
        f"{run_record['prompt_version']}, retrieval "
        f"{','.join(run_record['retrieval_methods']) or 'none'}, set "
        f"{run_record['dataset']} split {run_record['split']}, "
        f"{run_record['cases']} cases"
        + (f", {len(held_back)} held back" if held_back else "")
        + (
            f", judged by {judged_by['provider']}/{judged_by['model']} "
            f"(rubric {judged_by['rubric_version']})"
            if judged_by
            else ", not judged"
        )
    )
    for document, state in sorted(run_record["documents"].items()):
        print(
            f"  {document}: index generation {state['index_generation']}, "
            f"indexed in {state['indexed_seconds'] or 0.0:.2f}s"
        )
    print(
        f"Retrieval@{retrieval['k']}: hit {retrieval['hit_rate']:.2f}  "
        f"recall {retrieval['recall']:.2f}  mrr {retrieval['mrr']:.2f}  "
        f"ndcg {retrieval['ndcg']:.2f}  ({retrieval['questions']} questions)"
    )
    outcomes = {name: count for name, count in run_report["outcomes"].items() if count}
    if outcomes:
        print("Outcomes: " + ", ".join(f"{k}={v}" for k, v in outcomes.items()))
    answers = run_report["answers"]
    print(
        f"Answers: {answers['generated']} generated, {answers['truncated']} truncated, "
        "finish reasons "
        + (
            ", ".join(f"{k}={v}" for k, v in answers["finish_reasons"].items())
            or "none"
        )
    )
    for name in (
        "correctness",
        "faithfulness",
        "citation_precision",
        "citation_recall",
        "abstention",
    ):
        print(f"  {name}: {_mean(answers[name])}")
    for label, phase in (
        ("Latency (steady state)", "steady_state"),
        ("  cold start", "cold_start"),
    ):
        measured = run_report["latency"][phase]
        print(
            f"{label}: retrieval {_seconds(measured['retrieval_seconds'])}, first "
            f"token {_seconds(measured['first_token_seconds'])}, total "
            f"{_seconds(measured['total_seconds'])}"
        )
    cost = run_report["cost"]
    print(
        f"Cost: ${cost['usd']:.4f} over {cost['priced_cases']} answered cases "
        f"({cost['input_tokens']} in, {cost['output_tokens']} out) at "
        f"{cost['model']} ${cost['input_cost_per_million_usd']}/"
        f"${cost['output_cost_per_million_usd']} per million"
    )
    calibration = run_report.get("calibration")
    if calibration:
        agreement = calibration["agreement"]
        print(
            "Calibration: "
            + (
                f"{agreement:.0%} agreement on {calibration['agreed'] + calibration['disagreed']} "
                f"decided cases, {calibration['unknown']} unknown"
                if agreement is not None
                else f"the judge decided none of {calibration['cases']} labelled cases"
            )
        )
    for case in run_report["cases"]:
        line = f"  {case['id']}: {case['outcome']}"
        if case["hit_at_k"] is not None:
            line += (
                f" hit={case['hit_at_k']} recall={case['recall_at_k']:.2f}"
                f" mrr={case['reciprocal_rank']:.2f} ndcg={case['ndcg_at_k']:.2f}"
            )
        if case.get("verdict"):
            line += f" judge={case['verdict']}"
        if case.get("correctness_verdict"):
            line += f" correctness={case['correctness_verdict']}"
        failed = [
            name
            for name, grade in case["grades"].items()
            if grade["outcome"] == "failed"
        ]
        if failed:
            line += f" failed={','.join(failed)}"
        print(line)


def main(argv=None):
    """Parse the arguments and print the run."""
    global PROVIDER_PACE_SECONDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true", help="run against real providers"
    )
    parser.add_argument("--no-judge", action="store_true", help="skip the model judge")
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="skip grading the hand-labelled calibration set with the judge",
    )
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        help=f"chat provider that generates, default {DEFAULT_PROVIDER}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"chat model that generates, default {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--judge-provider",
        default=DEFAULT_PROVIDER,
        help=f"chat provider that judges, default {DEFAULT_PROVIDER}",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_MODEL,
        help=f"chat model that judges, default {DEFAULT_MODEL}",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--split",
        default=REPORTED,
        choices=SPLITS,
        help=f"which split of the case set to run, default {REPORTED}",
    )
    parser.add_argument("--json", dest="as_json", action="store_true")
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="force RERANK=true (local cross-encoder over 50 candidates)",
    )
    parser.add_argument(
        "--no-rerank", dest="rerank_off", action="store_true", help="force RERANK=false"
    )
    parser.add_argument(
        "--compare-rerank",
        action="store_true",
        help="run with and without reranking and report both",
    )
    parser.add_argument(
        "--pace",
        type=float,
        default=DEFAULT_PACE_SECONDS,
        help=(
            "seconds to wait between provider calls, so a rate-limited account "
            f"is not reported as a failing one; default {DEFAULT_PACE_SECONDS}, "
            "0 to send every call as fast as it can"
        ),
    )
    parser.add_argument(
        "--include-faults",
        action="store_true",
        help=(
            "ask the cases that need a fault injected too, which only reach "
            "their expected outcome in a run that injects it"
        ),
    )
    parser.add_argument(
        "--ablate",
        action="store_true",
        help=(
            "measure the declared retrieval experiments against the same cases "
            "instead of asking a model (no provider key needed)"
        ),
    )
    parser.add_argument(
        "--report",
        metavar="DIRECTORY",
        help="write a report — manifest, one file per experiment, and a summary",
    )
    parser.add_argument(
        "--compare-report",
        metavar="DIRECTORY",
        help="say whether this run reproduces the report in DIRECTORY, and what moved",
    )
    parser.add_argument(
        "--render",
        metavar="DIRECTORY",
        help=(
            "write the quality, latency, and cost digest for the report in "
            "DIRECTORY, reading it rather than measuring it again"
        ),
    )
    parser.add_argument(
        "--traces",
        metavar="FILE",
        action="append",
        default=[],
        help=(
            "a file of recorded answer traces to read the runtime half from; "
            "may be given more than once"
        ),
    )
    parser.add_argument(
        "--check",
        metavar="FILE",
        help=(
            "exit non-zero when the rendered digest crosses a bound declared in "
            "FILE; the digest is written either way"
        ),
    )
    parser.add_argument(
        "--brief-eval",
        action="store_true",
        help=(
            "grade the versioned Research Brief task set deterministically "
            "(no provider key needed)"
        ),
    )
    parser.add_argument(
        "--brief-report",
        metavar="DIRECTORY",
        help="write the brief tool-use report to DIRECTORY",
    )
    parser.add_argument(
        "--allow-paid",
        action="store_true",
        help=(
            "record new paid brief trajectories on the controlled schedule; "
            "refused without this flag"
        ),
    )
    parser.add_argument(
        "--brief-cost-limit",
        type=float,
        default=5.0,
        help="most one paid brief run may estimate before it refuses",
    )
    args = parser.parse_args(argv)
    for provider, model in (
        (args.provider, args.model),
        (args.judge_provider, args.judge_model),
    ):
        try:
            validate(provider, model)
        except SettingsError as exc:
            parser.error(str(exc))

    # Tri-state: None respects the configuration, True/False forces it.
    rerank = None
    if args.rerank:
        rerank = True
    elif args.rerank_off:
        rerank = False

    PROVIDER_PACE_SECONDS = max(args.pace, 0.0)
    if args.brief_eval:
        return _brief_eval_main(args)
    if args.render:
        return _render_main(args)
    if args.ablate:
        return _ablation_main(args)
    run_report = run(
        live=args.live,
        use_judge=not args.no_judge,
        k=args.k,
        provider=args.provider,
        model=args.model,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        rerank=rerank,
        compare_rerank=args.compare_rerank,
        calibrate=not args.no_calibration,
        split=args.split,
        include_faults=args.include_faults,
    )
    if args.report:
        _publish_report([result_of_run(run_report)], [run_report], args)
    if args.as_json:
        print(json.dumps(run_report, indent=2))
        return
    _print(run_report)


def _brief_eval_main(args) -> None:
    """
    Grade the brief task set deterministically and optionally publish it.

    Deterministic checks run routinely with no key: the trajectories are
    recorded fixtures, so grading them measures the graders rather than a
    provider. Recording new paid trajectories needs ``--allow-paid`` and a
    cost estimate inside ``--brief-cost-limit``; without both the schedule
    gate refuses before the first paid call.
    """
    from evaluation.brief import brief_eval
    from evaluation.brief.brief_tasks import load_brief_tasks

    task_set = load_brief_tasks()
    if args.allow_paid:
        estimate = brief_eval.estimate_cost_usd(len(task_set.tasks))
        brief_eval.check_paid_allowance(
            allow_paid=True,
            estimated_cost_usd=estimate,
            limit_usd=args.brief_cost_limit,
        )
        print(
            "Paid brief recording is gated to a controlled schedule; "
            f"estimate ${estimate:.2f} inside ${args.brief_cost_limit:.2f}. "
            "This checkout records deterministic checks only."
        )
    trajectories = brief_eval.demo_trajectories(task_set)
    evaluation = brief_eval.grade_trials(task_set, trajectories)
    if args.brief_report:
        brief_eval.write_brief_report(
            Path(args.brief_report),
            task_set=task_set,
            evaluation=evaluation,
            trajectories=trajectories,
        )
        print(f"Brief report written to {args.brief_report}")
    if args.as_json:
        print(json.dumps(evaluation.to_dict(), indent=2))
        return
    _print_brief_evaluation(evaluation)


def _print_brief_evaluation(evaluation) -> None:
    """Print one brief evaluation as a short operator-facing summary."""
    totals = evaluation.totals
    print(
        f"Brief tool use: {totals['trials']} trials, "
        f"p50 {totals['latency_p50']}s p95 {totals['latency_p95']}s, "
        f"${totals['cost_usd']:.4f}, timeouts {totals['timeout_rate']:.0%}"
    )
    for row in evaluation.configurations:
        print(
            f"  {row['configuration']}: {row['trials']} trials, "
            f"coverage {row['evidence_coverage']}, correctness {row['correctness']}"
        )
    for task in evaluation.tasks:
        print(
            f"  {task.task_id}: consistency {task.outcome_consistency:.0%}, "
            f"coverage {task.grade_pass_rate.get('evidence_coverage')}"
        )


def _render_main(args) -> None:
    """
    Re-read a report from disk, write its digest, and hold it to its thresholds.

    Nothing is measured here, which is the point: a report a reviewer has to
    pay a provider to look at again is a report nobody looks at again. The
    trace summary the digest's runtime half is built from is stored in the
    report directory, so a second render of the same report produces the same
    digest without the raw export files being anywhere near it.
    """
    directory = Path(args.render)
    if args.traces:
        # Read the export files and keep the summary: the raw traces are the
        # operator's local record, and the summary is what the report can hold.
        (directory / TRACE_SUMMARY_NAME).write_text(
            json.dumps(traces.summarize(args.traces), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    measured = digest.digest(
        report.load_manifest(directory),
        list(report.load_results(directory).values()),
        traces=args.traces or None,
        trace_summary=_stored_trace_summary(directory),
    )
    digest.write(directory, measured)
    print(f"Digest written to {directory}")
    if args.check:
        _check_thresholds(measured, Path(args.check))


def _stored_trace_summary(directory: Path) -> dict | None:
    """
    Return the trace summary a previous render stored, or None when there is none.

    The export files are the operator's and are not committed; the summary is
    what the report needs, so it lives in the report. Reading it back is what
    makes a second render reproduce the first.
    """
    stored = directory / TRACE_SUMMARY_NAME
    if not stored.exists():
        return None
    return json.loads(stored.read_text(encoding="utf-8"))


def _check_thresholds(measured: dict, thresholds: Path) -> None:
    """
    Hold the digest to its declared bounds, and say which ones it crossed.

    The digest is already written by the time this runs, deliberately: the
    artifact is what a person opens when the gate fails, and a gate that threw
    it away would leave them with an error message and nothing to look at.

    A bound over a number this run did not measure is reported as such and
    fails the run, because a check that passes what it cannot read is not a
    check. Nothing else fails: an undeclared number is nobody's business.
    """
    bounds = json.loads(thresholds.read_text(encoding="utf-8"))
    crossed = digest.regressions(measured, bounds)
    if not crossed:
        print(f"Inside every threshold in {thresholds}")
        return
    print(f"Crossed {len(crossed)} threshold(s) in {thresholds}:")
    for row in crossed:
        print(f"  {row['threshold']}: {row['reason']}")
    sys.exit(1)


def _ablation_main(args) -> None:
    """Measure the declared experiments, print them, and publish a report."""
    dataset = load_dataset()
    results = run_ablations(dataset, args.live, k=args.k, split=args.split)
    if args.report:
        _publish_report(results, [], args)
    if args.as_json:
        print(json.dumps(results, indent=2))
        return
    _print_ablations(results)


def _publish_report(results, run_records, args) -> None:
    """Write the report, then say whether it reproduces a published one."""
    directory = publish(load_dataset(), results, args.report, run_records)
    print(f"Report written to {directory}")
    if args.compare_report:
        outcome = report.reproduction(directory, Path(args.compare_report))
        if outcome["reproduced"]:
            print(f"Reproduces {args.compare_report}: the setup matches")
        else:
            print(f"Does not reproduce {args.compare_report}:")
            for change in outcome["changes"]:
                print(f"  {change['field']}: {change['before']} -> {change['after']}")
        for change in outcome["model_revisions_changed"]:
            print(
                f"  model revision moved: {change['field']} "
                f"{change['before'] or '(unpinned)'} -> {change['after'] or '(unpinned)'}"
            )
        moved = sorted(
            outcome["measurement_deltas"], key=lambda row: -row["largest_move"]
        )
        if moved and moved[0]["largest_move"] > 0:
            print(
                "Retrieval numbers moved between the two runs, by at most "
                f"{moved[0]['largest_move']:.3f} ("
                f"{moved[0]['experiment']}), which is the run-to-run movement a "
                "smaller difference should be read against:"
            )
            for row in moved:
                if row["largest_move"] > 0:
                    print(f"  {row['experiment']}: {row['largest_move']:.3f}")


def _print_ablations(results: list[dict]) -> None:
    """Print the comparison of experiments as a table, from the report's columns."""
    print(
        f"{'experiment':<24} {'hit':>6} {'recall':>7} {'mrr':>6} {'ndcg':>6} "
        f"{'p50 (s)':>9} {'p95 (s)':>9}"
    )
    for column in report.columns(results):
        print(
            f"{column.id:<24} {column.hit_rate:>6.2f} {column.recall:>7.2f} "
            f"{column.mrr:>6.2f} {column.ndcg:>6.2f} {column.p50:>9.4f} "
            f"{column.p95:>9.4f}"
        )
    for result in results:
        for direction in ("regressions", "improvements"):
            for row in result["comparison"][direction]:
                print(
                    f"  {result['experiment']} {direction[:-1]}: {row['id']} "
                    f"ndcg {row['delta']['ndcg_at_k']:+.2f}"
                )


if __name__ == "__main__":
    main()
