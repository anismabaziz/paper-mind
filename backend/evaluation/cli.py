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

With generation and a model judge:
    export PAPERMIND_EVAL_GENERATOR_API_KEY=...
    export PAPERMIND_EVAL_JUDGE_API_KEY=...
    uv run python -m evaluation.cli --live

Qdrant local URL is ``http://localhost:6333`` on the host
(``http://qdrant:6333`` inside compose, via ``QDRANT_URL``).
"""

import argparse
import json
import os
import sys
import time
from collections.abc import Callable


import settings as settings_module
from providers import get_vector_index
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
from storage import get_storage

from evaluation.evaluator import DEFAULT_K, evaluate, load_fixture
from evaluation.harness import build_environment, remove_documents
from evaluation.judge import Judge, JudgeSettings

EVAL_PREFIX = "eval-"

#: Environment variables the run reads its keys from. Neither is ever accepted
#: as an argument, and neither is printed: the run record names the provider
#: and the model, which is what makes a result comparable.
GENERATOR_API_KEY_ENV = "PAPERMIND_EVAL_GENERATOR_API_KEY"
JUDGE_API_KEY_ENV = "PAPERMIND_EVAL_JUDGE_API_KEY"

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


def generator_factory():
    """Return the factory the run's generator is built through."""

    def factory(credentials: ChatCredentials):
        """Build the generator for the credentials the answer path supplies."""
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
            budget=model_definition.chat_budget(),
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
    """Return the callable that sends one rubric prompt to the judge's model."""
    judge_provider = _judge_provider(provider, model, api_key)

    def grade(prompt: str) -> str:
        """Return the judge's reply, refusing to grade a call that failed."""
        verdict = judge_provider.generate_response(JUDGE_INSTRUCTION, prompt)
        if verdict in (judge_provider.FALLBACK_ANSWER,) or is_context_fallback(verdict):
            # Reported without the provider's own words: a failure can quote the
            # request or the key back, and neither belongs in a report.
            raise RuntimeError("Evaluation judge failed: the provider call failed")
        return verdict

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


def _run_once(
    fixture: dict,
    app_settings: Settings,
    *,
    provider: str,
    model: str,
    generator_key: str,
    judge: Judge | None = None,
    k: int = DEFAULT_K,
    calibrate: bool = True,
) -> dict:
    """Index the fixture documents, run every case, and clean up after itself."""
    from services.embeddings.local_embeddings import LocalEmbeddingService
    from services.retrieval.reranker import RerankerService
    from services.retrieval.vector_service import VectorService

    settings_module.set_settings(app_settings)
    environment = build_environment(
        fixture,
        settings=app_settings,
        session_factory=None,
        storage=get_storage(),
        embedding_service=LocalEmbeddingService(app_settings.embedding.embedding_model),
        vector_service=VectorService(
            get_vector_index(),
            RerankerService(
                app_settings.rerank.rerank_model, enabled=app_settings.rerank.enabled
            ),
        ),
        chat_provider_factory=generator_factory(),
        documents_prefix=EVAL_PREFIX,
    )
    try:
        started = time.monotonic()
        report = evaluate(
            fixture,
            environment,
            provider=environment.provider(model_for(provider, model), generator_key),
            model=model_for(provider, model),
            judge=judge,
            k=k,
            calibrate=calibrate,
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

    fixture = load_fixture()
    base_settings = settings_module.get_settings()
    if compare_rerank:
        return {
            "rerank": {
                label: _run_once(
                    fixture,
                    with_rerank(base_settings, enabled),
                    provider=provider,
                    model=model,
                    generator_key=generator_key,
                    judge=grader,
                    k=k,
                    calibrate=calibrate,
                )
                for label, enabled in (("off", False), ("on", True))
            }
        }
    return _run_once(
        fixture,
        base_settings if rerank is None else with_rerank(base_settings, rerank),
        provider=provider,
        model=model,
        generator_key=generator_key,
        judge=grader,
        k=k,
        calibrate=calibrate,
    )


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


def _print(report: dict) -> None:
    """Print a run as a short operator-facing summary."""
    if "rerank" in report:
        for label, run_record in report["rerank"].items():
            print(f"--- rerank {label} ---")
            _print(run_record)
        return
    run_record = report["run"]
    retrieval = report["retrieval"]
    judged_by = run_record.get("judge")
    print(
        f"Run: {run_record['provider']}/{run_record['model']}, prompt "
        f"{run_record['prompt_version']}, retrieval "
        f"{','.join(run_record['retrieval_methods']) or 'none'}"
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
    outcomes = {name: count for name, count in report["outcomes"].items() if count}
    if outcomes:
        print("Outcomes: " + ", ".join(f"{k}={v}" for k, v in outcomes.items()))
    answers = report["answers"]
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
        measured = report["latency"][phase]
        print(
            f"{label}: retrieval {_seconds(measured['retrieval_seconds'])}, first "
            f"token {_seconds(measured['first_token_seconds'])}, total "
            f"{_seconds(measured['total_seconds'])}"
        )
    cost = report["cost"]
    print(
        f"Cost: ${cost['usd']:.4f} over {cost['priced_cases']} answered cases "
        f"({cost['input_tokens']} in, {cost['output_tokens']} out) at "
        f"{cost['model']} ${cost['input_cost_per_million_usd']}/"
        f"${cost['output_cost_per_million_usd']} per million"
    )
    calibration = report.get("calibration")
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
    for case in report["cases"]:
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

    report = run(
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
    )
    if args.as_json:
        print(json.dumps(report, indent=2))
        return
    _print(report)


if __name__ == "__main__":
    main()
