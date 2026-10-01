# Retrieval baseline: case set 2026-09-labeled-cases-v1

Measured on revision `d85d2514c9ac` (fix: report how far a re-run's numbers moved, not only whether the setup matched)
Case set 2026-09-labeled-cases-v1 (reviewed 2026-09-27), split `tuning`: 16 questions asked of each of the 11 experiments, 1 of the split's cases held back (`skating-failure-provider`).
Embedding `BAAI/bge-m3`
Reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`
Citation prompt `grounded-claims-v1`, Python 3.11.13, RRF k=60, candidate depth 50.

## What this run measured

- **deterministic**: measured in `baseline`, `dense-only`, `sparse-only`, `hybrid-only`, `rerank-on`, `depth-10`, `depth-100`, `expansion-none`, `expansion-deterministic`, `chunk-256-25`, `chunk-1024-100`
- **model graded**: not measured — retrieval-only measurement, nothing asked a model
- **human calibration**: not measured — retrieval-only measurement, nothing asked a model
- **provider failures**: not measured — no provider was called, so no call could fail
- **answers**: not measured — retrieval-only measurement, nothing asked a model
- **citations**: not measured — retrieval-only measurement, nothing asked a model
- **abstention**: not measured — retrieval-only measurement, nothing asked a model
- **tokens**: not measured — retrieval-only measurement, nothing asked a model
- **cost**: not measured — retrieval-only measurement, nothing asked a model

## Experiments

This is the tuning half: the split every variant's value was chosen on. A number here is a decision, not a result — the reported half is a separate run, and no variant's value was changed after reading it.

| Experiment | Family | Changes | Hit@k | Recall | MRR | nDCG | Retrieval p50 (s) | Retrieval p95 (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline` | baseline | - | 0.812 | 0.812 | 0.693 | 0.693 | 0.0138 | 0.0370 |
| `dense-only` | retrieval-method | method=dense | 0.812 | 0.812 | 0.740 | 0.734 | 0.0106 | 0.0401 |
| `sparse-only` | retrieval-method | method=sparse | 0.812 | 0.781 | 0.740 | 0.717 | 0.0054 | 0.0107 |
| `hybrid-only` | retrieval-method | method=hybrid | 0.812 | 0.812 | 0.693 | 0.693 | 0.0096 | 0.0135 |
| `rerank-on` | rerank | rerank=True | 0.938 | 0.906 | 0.865 | 0.836 | 0.1351 | 0.9239 |
| `depth-10` | candidate-depth | candidate_depth=10 | 0.812 | 0.812 | 0.724 | 0.716 | 0.0120 | 0.0192 |
| `depth-100` | candidate-depth | candidate_depth=100 | 0.812 | 0.812 | 0.693 | 0.693 | 0.0103 | 0.0135 |
| `expansion-none` | query-expansion | query_expansion=none | 0.812 | 0.812 | 0.724 | 0.716 | 0.0094 | 0.0141 |
| `expansion-deterministic` | query-expansion | query_expansion=deterministic | 0.875 | 0.875 | 0.705 | 0.718 | 0.0094 | 0.0138 |
| `chunk-256-25` | chunking | chunk_size_tokens=256, chunk_overlap_tokens=25 | 0.875 | 0.875 | 0.762 | 0.766 | 0.0110 | 0.0198 |
| `chunk-1024-100` | chunking | chunk_size_tokens=1024, chunk_overlap_tokens=100 | 0.938 | 0.906 | 0.773 | 0.760 | 0.0089 | 0.0257 |

## Questions that moved

| Experiment | Direction | Question | nDCG change |
| --- | --- | --- | --- |
| `dense-only` | regressions | `skating-jump-counts` | -0.500 |
| `dense-only` | regressions | `skating-candidate-counts` | -0.296 |
| `dense-only` | improvements | `skating-sensor-placement` | 0.500 |
| `dense-only` | improvements | `skating-participants` | 0.569 |
| `dense-only` | improvements | `eval-metrics-list` | 0.369 |
| `sparse-only` | regressions | `skating-sensor-placement` | -0.500 |
| `sparse-only` | regressions | `skating-participants` | -1.000 |
| `sparse-only` | regressions | `eval-metrics-list` | -0.369 |
| `sparse-only` | improvements | `skating-jump-counts` | 1.000 |
| `sparse-only` | improvements | `skating-candidate-counts` | 0.296 |
| `sparse-only` | improvements | `skating-pps-across-sections` | 0.307 |
| `hybrid-only` | regressions | `skating-jump-counts` | -0.500 |
| `hybrid-only` | regressions | `skating-pps-across-sections` | -0.307 |
| `hybrid-only` | improvements | `skating-participants` | 0.431 |
| `rerank-on` | regressions | `skating-candidate-counts` | -0.043 |
| `rerank-on` | improvements | `skating-sensor-placement` | 1.000 |
| `rerank-on` | improvements | `skating-participants` | 0.569 |
| `rerank-on` | improvements | `skating-pps-across-sections` | 0.387 |
| `rerank-on` | improvements | `eval-metrics-list` | 0.369 |
| `depth-10` | regressions | `skating-sensor-placement` | -1.000 |
| `depth-10` | regressions | `skating-participants` | -0.569 |
| `depth-10` | regressions | `skating-pps-across-sections` | -0.387 |
| `depth-10` | improvements | `skating-candidate-counts` | 0.043 |
| `depth-100` | regressions | `eval-metrics-list` | -0.369 |
| `expansion-none` | improvements | `eval-metrics-list` | 0.369 |
| `expansion-deterministic` | regressions | `eval-metrics-list` | -0.369 |
| `expansion-deterministic` | improvements | `skating-sensor-placement` | 0.387 |
| `chunk-256-25` | regressions | `skating-participants` | -0.044 |
| `chunk-256-25` | improvements | `skating-jump-counts` | 0.131 |
| `chunk-256-25` | improvements | `skating-sensor-placement` | 0.244 |
| `chunk-256-25` | improvements | `skating-candidate-counts` | 0.080 |
| `chunk-256-25` | improvements | `eval-metrics-list` | 0.369 |
| `chunk-1024-100` | regressions | `skating-jump-counts` | -0.244 |
| `chunk-1024-100` | regressions | `skating-sensor-placement` | -0.131 |
| `chunk-1024-100` | regressions | `skating-candidate-counts` | -0.080 |
| `chunk-1024-100` | regressions | `eval-metrics-list` | -0.369 |
| `chunk-1024-100` | improvements | `skating-participants` | 0.113 |
| `chunk-1024-100` | improvements | `skating-pps-across-sections` | 0.613 |

## Reproducing this report

```
docker compose up db qdrant
uv run python -m evaluation.cli --live --report <directory> --ablate
uv run python -m evaluation.cli --live --report <directory> --compare-report <published directory>
```

The manifest holds the revision, the case set, the document hashes, the prompts, the models, the settings, and the environment; each result beside it holds the index manifest and generation of every document, with the per-question rows the numbers came from. A re-run says whether the setup matched and how far each experiment's numbers moved, and a changed model revision is named on its own.

The setup is checkable; the numbers are a sample. Retrieval is measured against an approximate index whose sparse scores carry term weights taken across the whole collection, so two runs over the same documents in the same collection can rank a borderline passage differently. Treat a difference smaller than the reported movement as no difference.
