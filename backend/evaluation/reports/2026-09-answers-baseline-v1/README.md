# Answers: case set 2026-09-labeled-cases-v1

Measured on revision `f49885ce36c4` (feat: Keep PDF reading within browser limits), with uncommitted changes
Case set 2026-09-labeled-cases-v1 (reviewed 2026-09-27), split `validation`: 37 questions asked of the shipped configuration. 3 of the split's cases held back (`eval-failure-citation`, `notes-failure-provider`, `skating-failure-citation`).
Embedding `BAAI/bge-m3` at `5617a9f61b028005a4858fdac845db406aefb181`
Generating model `qwen/qwen3.8-27b` on groq
Judging model `openai/gpt-oss-20b` on groq

Citation prompt `grounded-claims-v1`, Python 3.11.13, RRF k=60, candidate depth 50.

## What this run measured

- **deterministic**: measured in `baseline`
- **model graded**: measured in `baseline`
- **human calibration**: measured in `baseline`
- **provider failures**: measured in `baseline`
- **answers**: measured in `baseline`
- **citations**: measured in `baseline`
- **abstention**: measured in `baseline`
- **tokens**: measured in `baseline`
- **cost**: measured in `baseline`

## Answers

| Metric | Mean | Scored | Unknown | Failed |
| --- | --- | --- | --- | --- |
| Correctness | 0.794 | 34/37 | 3 | 7 |
| Faithfulness | 0.956 | 34/37 | 3 | 2 |
| Citation precision | 0.994 | 26/37 | 11 | 1 |
| Citation recall | 1.000 | 26/37 | 11 | 0 |
| Abstention | 0.906 | 32/37 | 5 | 3 |
| Cost | $0.0724 | 34 answered | 65374 in / 5013 out | - |

The judge agreed with the hand-labelled set on 75% of 4 decided cases.

## Cases that failed

| Question | Outcome | Failed graders | Judge | Correctness |
| --- | --- | --- | --- | --- |
| `skating-isolated-jumps` | citation_error | provider_status,required_abstention | - | - |
| `skating-errors` | answered | exact_evidence | - | - |
| `skating-rotation-threshold` | answered | exact_evidence | - | incorrect |
| `skating-table-successful-pps` | answered | exact_evidence | unfaithful | incorrect |
| `eval-sparse-paradox` | citation_error | provider_status,required_abstention | - | - |
| `eval-what-cannot-be-generated` | answered | exact_evidence | - | incorrect |
| `eval-report-alongside` | answered | exact_evidence | - | - |
| `notes-injection-not-agreed` | citation_error | provider_status,required_abstention | - | - |
| `skating-rotation-speed` | answered | exact_evidence | - | incorrect |

## Reproducing this report

```
docker compose up db qdrant
export PAPERMIND_EVAL_GENERATOR_API_KEY=...
export PAPERMIND_EVAL_JUDGE_API_KEY=...
uv run python -m evaluation.cli --live \
  --provider groq --model qwen/qwen3.8-27b \
  --judge-provider groq --judge-model openai/gpt-oss-20b \
  --report <directory>
```

The manifest holds the revision, the case set, the document hashes, the prompts, the models, the settings, and the environment; each result beside it holds the index manifest and generation of every document, with the per-question rows the numbers came from. A re-run says whether the setup matched and how far each experiment's numbers moved, and a changed model revision is named on its own.

The setup is checkable; the numbers are a sample. Retrieval is measured against an approximate index whose sparse scores carry term weights taken across the whole collection, so two runs over the same documents in the same collection can rank a borderline passage differently. Treat a difference smaller than the reported movement as no difference.
