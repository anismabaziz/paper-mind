# Quality, latency, and cost: case set 2026-09-labeled-cases-v1

Split `tuning`, 16 questions, on revision `d85d2514c9ac` — fix: report how far a re-run's numbers moved, not only whether the setup matched
No model generated.
Citation prompt `grounded-claims-v1`, embedding `BAAI/bge-m3`

These three answers come from the same cases. A configuration cannot look cheap here by being asked fewer questions, and cannot look fast by failing the slow ones: the failures are in the quality table and the abstentions are counted as results.

## Quality

- **abstention**: not measured — retrieval-only measurement, nothing asked a model
- **answers**: not measured — retrieval-only measurement, nothing asked a model
- **citations**: not measured — retrieval-only measurement, nothing asked a model

This run measured no answers, so it has no answer, citation, or abstention quality to report.

Retrieval over 16 questions: hit@k 0.812, recall 0.812, MRR 0.693, nDCG 0.693.

What did not go right: nothing; graders that rejected an answer: none. The 16 cases behind those numbers are listed in the report beside this one.

## Performance

Seconds, as p50 / p95. A p50 alone is the answer most readers get and a p95 alone is the one they complain about.

| Where | Retrieval | First token | Total |
| --- | --- | --- | --- |
| run, cold start | no samples | no samples | no samples |
| run, steady | 0.01 / 0.04 | no samples | no samples |
| recorded requests | - | - | no recorded traces were read for this report |

## Configurations

Change against the shipped configuration, which is compared with itself so the other rows read as differences rather than as absolutes.

| Configuration | nDCG | Correctness | Retrieval p50 (s) | Total p50 (s) | Total p95 (s) | Cost ($) | Cost a case ($) | Indexing (s) | What did not go right |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline (reference)` | +0.000 | - | +0.000 | - | - | - | - | +0.000 | nothing |
| `chunk-1024-100` | +0.067 | - | -0.005 | - | - | - | - | +4.638 | nothing |
| `chunk-256-25` | +0.073 | - | -0.003 | - | - | - | - | -0.882 | nothing |
| `dense-only` | +0.040 | - | -0.003 | - | - | - | - | +0.000 | nothing |
| `depth-10` | +0.023 | - | -0.002 | - | - | - | - | +0.000 | nothing |
| `depth-100` | +0.000 | - | -0.003 | - | - | - | - | +0.000 | nothing |
| `expansion-deterministic` | +0.024 | - | -0.004 | - | - | - | - | +0.000 | nothing |
| `expansion-none` | +0.023 | - | -0.004 | - | - | - | - | +0.000 | nothing |
| `hybrid-only` | +0.000 | - | -0.004 | - | - | - | - | +0.000 | nothing |
| `rerank-on` | +0.143 | - | +0.121 | - | - | - | - | +0.000 | nothing |
| `sparse-only` | +0.023 | - | -0.008 | - | - | - | - | +0.000 | nothing |

## Cost

No model was called, so this run was billed nothing. The provider, the tokens, and the dollars are absent from this report rather than reported as zero, because a run that asked nobody is not a run that asked cheaply.

Local compute is declared, not billed: embedding `BAAI/bge-m3` and reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2` were not applied, on cpu, took 40.1s to index 4 documents (10.0s a document).

## Reproducing this digest

```
docker compose up qdrant
uv run python -m evaluation.cli --live --ablate --report <directory>
uv run python -m evaluation.cli --render <directory>
```
