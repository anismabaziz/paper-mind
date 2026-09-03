"""Opt-in live evaluation CLI.

Without ``--live`` this refuses to run: the point of the evaluator is a
measured number, but a live run costs provider calls and writes into the
real vector index, so it only ever happens on purpose.

    python -m evaluation.cli --live            # full run with LLM judge
    python -m evaluation.cli --live --no-judge # retrieval metrics only
"""

import argparse
import json
import sys

import config
from evaluation.evaluator import (
    DEFAULT_K,
    evaluate,
    index_document,
    load_fixture,
    remove_document,
)

EVAL_PREFIX = "eval-"


def make_live_components():
    """Wire the evaluator to the real providers via config's lazy clients."""
    # Reuse the app's own embedding and generation paths (provider,
    # fallback, prompt) so the numbers describe what users actually get.
    from services.ai_service import AIService

    embed_fn = AIService.get_embeddings

    def generate_fn(query, context):
        return AIService.generate_response(query, context)

    def judge_fn(prompt):
        from google.genai import types

        result = config.genai_client.models.generate_content(
            model=config.CHAT_MODEL,
            config=types.GenerateContentConfig(
                system_instruction="You are a strict evaluation judge. Follow the output format exactly."
            ),
            contents=[prompt],
        )
        return result.text or ""

    return embed_fn, generate_fn, judge_fn


def run(live: bool, judge: bool, k: int) -> dict:
    if not live:
        sys.exit(
            "Refusing to run a live evaluation by default. Add --live to "
            "embed, retrieve, generate, and judge against the real providers."
        )

    fixture = load_fixture()
    config.validate()
    embed_fn, generate_fn, judge_fn = make_live_components()
    if not judge:
        judge_fn = None

    index = config.vector_index
    try:
        for doc in fixture["documents"]:
            name = f"{EVAL_PREFIX}{doc['filename']}"
            chunks = index_document(
                doc["filename"], index, embed_fn, pdf_name=name
            )
            print(f"indexed {name}: {chunks} chunks")
        report = evaluate(
            fixture, index, embed_fn, generate_fn, judge_fn, k=k, prefix=EVAL_PREFIX
        )
    finally:
        for doc in fixture["documents"]:
            remove_document(f"{EVAL_PREFIX}{doc['filename']}", index)

    return report.as_dict()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="run against real providers")
    parser.add_argument(
        "--no-judge", action="store_true", help="skip LLM-as-judge faithfulness"
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--json", dest="as_json", action="store_true")
    args = parser.parse_args(argv)

    report = run(live=args.live, judge=not args.no_judge, k=args.k)
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        r = report["retrieval"]
        f = report["faithfulness"]
        print(f"Retrieval: hit@{r['k']} {r['hit_rate']:.2f}  recall@{r['k']} {r['recall']:.2f}  ({r['questions']} questions)")
        if f["judged"]:
            print(f"Faithfulness: {f['mean']:.2f} mean ({f['faithful']}/{f['judged']} fully faithful)")
        for q in report["per_question"]:
            line = f"  {q['id']}: hit={q['hit_at_k']} recall={q['recall_at_k']:.2f}"
            if "verdict" in q:
                line += f"  judge={q['verdict']}"
            print(line)


if __name__ == "__main__":
    main()
