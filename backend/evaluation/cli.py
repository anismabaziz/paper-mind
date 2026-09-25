"""
Opt-in live evaluation CLI.

Without ``--live`` this refuses to run: the point of the evaluator is a
measured number, but a live run writes into the real vector index, so it
only ever happens on purpose.

Free local (no keys, default):
docker compose up qdrant                          # or `docker compose up` (exposes http://localhost:6333)
uv run python -m evaluation.cli --live --no-judge  # hit@5/recall@5 only, no LLM

With generation + LLM judge (needs a chat key):
uv run python -m evaluation.cli --live --provider google --model gemini-2.5-flash --api-key ...

Qdrant local URL is ``http://localhost:6333`` on the host
(``http://qdrant:6333`` inside compose, via ``QDRANT_URL``).
"""

import argparse
import json
import sys

import settings
from providers import get_vector_index
from services.accounts.chat_settings_service import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    SettingsError,
    model_for,
    validate,
)

from evaluation.evaluator import (
    DEFAULT_K,
    evaluate,
    index_document,
    load_fixture,
    remove_document,
)

EVAL_PREFIX = "eval-"


def _validate_judge_provider(provider: str, judge: bool) -> None:
    """Reject a judge run that would use a different provider."""
    if judge and provider != "google":
        raise SettingsError(
            "LLM-as-judge currently supports Google only. Use --no-judge "
            "for a Groq generation run."
        )


def make_live_components(
    provider: str,
    model: str,
    api_key: str,
    judge_enabled: bool = True,
):
    """Wire the evaluator to the chosen per-run provider settings."""
    validate(provider, model)
    _validate_judge_provider(provider, judge_enabled)
    model_definition = model_for(provider, model)
    # Reuse the app's own embedding and generation paths (prompt, factory)
    # so the numbers describe what users actually get.
    from settings import get_settings

    from services.embeddings.local_embeddings import LocalEmbeddingService
    from services.llm.base import ChatCredentials
    from services.llm.factory import build_chat_provider
    from google import genai
    from google.genai import types

    embed_fn = LocalEmbeddingService(
        get_settings().embedding.embedding_model
    ).embed_texts

    chat_provider = (
        build_chat_provider(
            ChatCredentials(
                provider=provider,
                model=model,
                api_key=api_key,
                verification_timeout_seconds=model_definition.timeout_seconds,
            ),
            use_cache=False,
        )
        if api_key
        else None
    )

    def generate_fn(query, context, prior_turns=""):
        """Do generate fn."""
        if chat_provider is None:
            return ""
        return chat_provider.generate_response(query, context, prior_turns)

    def judge_fn(prompt):
        """Do judge fn."""
        if not api_key:
            return "unparseable"
        try:
            result = genai.Client(api_key=api_key).models.generate_content(
                model=model,
                config=types.GenerateContentConfig(
                    system_instruction="You are a strict evaluation judge. Follow the output format exactly."
                ),
                contents=[prompt],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Evaluation judge failed: {type(exc).__name__}"
            ) from None
        return result.text or ""

    return embed_fn, generate_fn, judge_fn


def run(
    live: bool,
    judge: bool,
    k: int,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    api_key: str = "",
    rerank=None,
    compare_rerank: bool = False,
    **kwargs,
) -> dict:
    """Do run."""
    if not live:
        sys.exit(
            "Refusing to run a live evaluation by default. Add --live to "
            "embed, retrieve, generate, and judge against the real providers."
        )
    validate(provider, model)
    if judge and not api_key:
        sys.exit(
            "LLM generation and judging need an API key. Pass --api-key with "
            "the key for the chosen --provider, or run with --no-judge."
        )
    try:
        _validate_judge_provider(provider, judge)
    except SettingsError as exc:
        sys.exit(str(exc))

    fixture = load_fixture()
    settings.validate()
    embed_fn, generate_fn, judge_fn = make_live_components(
        provider,
        model,
        api_key,
        judge_enabled=judge,
    )
    if not judge:
        judge_fn = None

    index = get_vector_index()
    try:
        ingest_times: list[float] = []
        for doc in fixture["documents"]:
            name = f"{EVAL_PREFIX}{doc['filename']}"
            from evaluation.evaluator import index_document_timed

            chunks, elapsed = index_document_timed(
                doc["filename"], index, embed_fn, pdf_name=name
            )
            ingest_times.append(elapsed)
            print(
                f"indexed {name}: {chunks} chunks in {elapsed:.2f}s ({elapsed:.2f}s/PDF)"
            )
        if ingest_times:
            avg = sum(ingest_times) / len(ingest_times)
            print(
                f"Ingest: {sum(ingest_times):.2f}s total, {avg:.2f}s/PDF over {len(ingest_times)} sample_docs (hit@5/recall@5 gate via evaluator.py)"
            )

        if compare_rerank:
            from evaluation.evaluator import evaluate_with_rerank_comparison

            result = evaluate_with_rerank_comparison(
                fixture, index, embed_fn, generate_fn, judge_fn, k=k, prefix=EVAL_PREFIX
            )
            # Return the reranked report for the JSON output, but keep both
            return result["on"] if isinstance(result.get("on"), dict) else result

        # rerank=None respects Settings; True/False forces it
        report = evaluate(
            fixture,
            index,
            embed_fn,
            generate_fn,
            judge_fn,
            k=k,
            prefix=EVAL_PREFIX,
            rerank=rerank,
        )
    finally:
        for doc in fixture["documents"]:
            remove_document(f"{EVAL_PREFIX}{doc['filename']}", index)

    return report.as_dict()


def main(argv=None):
    """Do main."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true", help="run against real providers"
    )
    parser.add_argument(
        "--no-judge", action="store_true", help="skip LLM-as-judge faithfulness"
    )
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        help="chat provider for generation (google|groq), default google",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"chat model for generation, default {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="API key for the chosen provider (required unless --no-judge)",
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
        help="run with and without reranking and log hit@k/faithfulness + latency delta for 50 docs",
    )
    args = parser.parse_args(argv)
    try:
        validate(args.provider, args.model)
    except SettingsError as exc:
        parser.error(str(exc))

    # Tri-state: None respects env, True/False forces
    rerank = None
    if args.rerank:
        rerank = True
    elif getattr(args, "rerank_off", False):
        rerank = False

    # Backward-compat: tests monkeypatch run with lambda live,judge,k only
    try:
        report = run(
            live=args.live,
            judge=not args.no_judge,
            k=args.k,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            rerank=rerank,
            compare_rerank=args.compare_rerank,
        )
    except TypeError as exc:
        if "unexpected keyword" in str(exc):
            report = run(live=args.live, judge=not args.no_judge, k=args.k)
        else:
            raise
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        r = report["retrieval"]
        f = report["faithfulness"]
        print(
            f"Retrieval: hit@{r['k']} {r['hit_rate']:.2f}  recall@{r['k']} {r['recall']:.2f}  ({r['questions']} questions)"
        )
        if f["judged"]:
            print(
                f"Faithfulness: {f['mean']:.2f} mean ({f['faithful']}/{f['judged']} fully faithful)"
            )
        for q in report["per_question"]:
            line = f"  {q['id']}: hit={q['hit_at_k']} recall={q['recall_at_k']:.2f}"
            if "verdict" in q:
                line += f"  judge={q['verdict']}"
            print(line)


if __name__ == "__main__":
    main()
