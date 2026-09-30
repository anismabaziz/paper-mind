# Quality, latency, and cost: case set 2026-09-labeled-cases-v1

Split `validation`, 37 questions, on revision `f49885ce36c4`, with uncommitted changes — feat: Keep PDF reading within browser limits
Generating model `qwen/qwen3.8-27b` on groq, judged by `openai/gpt-oss-20b` on groq (rubric faithfulness-rubric-v2)
Citation prompt `grounded-claims-v1`, embedding `BAAI/bge-m3` at `5617a9f61b028005a4858fdac845db406aefb181`

These three answers come from the same cases. A configuration cannot look cheap here by being asked fewer questions, and cannot look fast by failing the slow ones: the failures are in the quality table and the abstentions are counted as results.

## Quality


| Kind | Measure | Mean | Scored | Graded | Unknown | Failed |
| --- | --- | --- | --- | --- | --- | --- |
| answers | correctness | 0.794 | 34 | 37 | 3 | 7 |
| answers | faithfulness | 0.956 | 34 | 37 | 3 | 2 |
| citations | citation precision | 0.994 | 26 | 37 | 11 | 1 |
| citations | citation recall | 1.000 | 26 | 37 | 11 | 0 |
| abstention | abstention | 0.906 | 32 | 37 | 5 | 3 |

Retrieval over 37 questions: hit@k 0.730, recall 0.730, MRR 0.668, nDCG 0.629.

34 cases were answered. What did not go right: citation_error 3; graders that rejected an answer: provider_status 3, required_abstention 3, exact_evidence 6. The 9 cases behind those numbers are listed in the report beside this one.

## Performance

Seconds, as p50 / p95. A p50 alone is the answer most readers get and a p95 alone is the one they complain about.

| Where | Retrieval | First token | Total |
| --- | --- | --- | --- |
| run, cold start | 4.37 / 4.37 | 0.92 / 0.92 | 1.13 / 1.13 |
| run, steady | 4.03 / 4.51 | 0.53 / 1.45 | 0.96 / 2.13 |
| recorded requests | 4.26 / 6.62 | 0.53 / 1.10 | 0.83 / 2.02 |

What those 158 requests ended as: answered 122, citation_error 7, provider_error 29.
They arrived over 26.7 hours, which is 0.10 a minute.

## Cost

Provider: 65374 input and 5013 output tokens over 34 answered cases at `qwen/qwen3.8-27b`, $0.0724 in total and $0.0021 per answered case.
Recorded requests outside the case set: 291134 input and 19741 output tokens over 129 calls at `qwen/qwen3.8-27b`, $0.2699 in total and $0.0021 a call.

Local compute is declared, not billed: embedding `BAAI/bge-m3` and reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2` were not applied, on cpu, took 71.0s to index 4 documents (17.7s a document).

## Reproducing this digest

```
docker compose up db qdrant
export PAPERMIND_EVAL_GENERATOR_API_KEY=...
export PAPERMIND_EVAL_JUDGE_API_KEY=...
uv run python -m evaluation.cli --live \
  --provider groq --model qwen/qwen3.8-27b \
  --judge-provider groq --judge-model openai/gpt-oss-20b \
  --report <directory>
uv run python -m evaluation.cli --render <directory>
```
