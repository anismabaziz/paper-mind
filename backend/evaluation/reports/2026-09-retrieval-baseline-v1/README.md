# Retrieval baseline: case set 2026-09-labeled-cases-v1

Measured on revision `97d4ceb28e3d` (fix: say which split a report is read on)
Case set 2026-09-labeled-cases-v1 (reviewed 2026-09-27), split `validation`: 37 questions asked of each of the 11 experiments, 3 of the split's cases held back (`eval-failure-citation`, `notes-failure-provider`, `skating-failure-citation`).
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

Each variant's value was chosen on the `tuning` split, never on the split reported here.

| Experiment | Family | Changes | Hit@k | Recall | MRR | nDCG | Retrieval p50 (s) | Retrieval p95 (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline` | baseline | - | 0.730 | 0.730 | 0.641 | 0.613 | 0.0167 | 0.0403 |
| `dense-only` | retrieval-method | method=dense | 0.730 | 0.716 | 0.598 | 0.575 | 0.0123 | 0.0199 |
| `sparse-only` | retrieval-method | method=sparse | 0.757 | 0.757 | 0.698 | 0.663 | 0.0116 | 0.0232 |
| `hybrid-only` | retrieval-method | method=hybrid | 0.730 | 0.730 | 0.641 | 0.613 | 0.0142 | 0.0401 |
| `rerank-on` | rerank | rerank=True | 0.757 | 0.757 | 0.637 | 0.626 | 0.3013 | 1.1006 |
| `depth-10` | candidate-depth | candidate_depth=10 | 0.730 | 0.730 | 0.641 | 0.615 | 0.0403 | 0.0678 |
| `depth-100` | candidate-depth | candidate_depth=100 | 0.730 | 0.730 | 0.641 | 0.615 | 0.0571 | 0.0890 |
| `expansion-none` | query-expansion | query_expansion=none | 0.730 | 0.730 | 0.654 | 0.625 | 0.0584 | 0.0866 |
| `expansion-deterministic` | query-expansion | query_expansion=deterministic | 0.730 | 0.730 | 0.641 | 0.613 | 0.0560 | 0.0831 |
| `chunk-256-25` | chunking | chunk_size_tokens=256, chunk_overlap_tokens=25 | 0.757 | 0.757 | 0.606 | 0.595 | 0.0151 | 0.0338 |
| `chunk-1024-100` | chunking | chunk_size_tokens=1024, chunk_overlap_tokens=100 | 0.784 | 0.784 | 0.671 | 0.655 | 0.0108 | 0.0370 |

## Questions that moved

| Experiment | Direction | Question | nDCG change |
| --- | --- | --- | --- |
| `dense-only` | regressions | `skating-missed-jumps` | -0.080 |
| `dense-only` | regressions | `skating-height-underestimate` | -0.237 |
| `dense-only` | regressions | `skating-conclusion` | -0.569 |
| `dense-only` | regressions | `eval-correctness` | -0.369 |
| `dense-only` | regressions | `eval-abstention-errors` | -0.131 |
| `sparse-only` | regressions | `skating-false-positive` | -0.500 |
| `sparse-only` | regressions | `skating-table-worst-flight-time` | -0.387 |
| `sparse-only` | improvements | `skating-errors` | 1.000 |
| `sparse-only` | improvements | `skating-missed-jumps` | 0.307 |
| `sparse-only` | improvements | `skating-height-underestimate` | 0.080 |
| `sparse-only` | improvements | `skating-conclusion` | 0.569 |
| `sparse-only` | improvements | `eval-correctness` | 0.369 |
| `sparse-only` | improvements | `eval-citation-pair` | 0.226 |
| `sparse-only` | improvements | `eval-abstention-errors` | 0.500 |
| `sparse-only` | improvements | `eval-what-cannot-be-generated` | 0.080 |
| `sparse-only` | improvements | `skating-rotation-speed` | 1.000 |
| `hybrid-only` | regressions | `skating-errors` | -1.000 |
| `hybrid-only` | regressions | `skating-missed-jumps` | -0.226 |
| `hybrid-only` | regressions | `eval-citation-pair` | -0.226 |
| `hybrid-only` | regressions | `eval-abstention-errors` | -0.369 |
| `hybrid-only` | regressions | `eval-what-cannot-be-generated` | -0.080 |
| `hybrid-only` | regressions | `skating-rotation-speed` | -1.000 |
| `hybrid-only` | improvements | `skating-false-positive` | 0.500 |
| `hybrid-only` | improvements | `skating-table-worst-flight-time` | 0.387 |
| `hybrid-only` | improvements | `skating-height-underestimate` | 0.157 |
| `rerank-on` | regressions | `skating-missed-jumps` | -0.123 |
| `rerank-on` | regressions | `skating-table-worst-flight-time` | -0.387 |
| `rerank-on` | regressions | `skating-rotation-doubles-triples` | -0.369 |
| `rerank-on` | improvements | `skating-errors` | 0.500 |
| `rerank-on` | improvements | `skating-height-underestimate` | 0.150 |
| `rerank-on` | improvements | `eval-what-cannot-be-generated` | 0.080 |
| `rerank-on` | improvements | `skating-rotation-speed` | 0.631 |
| `depth-10` | regressions | `skating-errors` | -0.500 |
| `depth-10` | regressions | `skating-height-underestimate` | -0.150 |
| `depth-10` | regressions | `skating-rotation-speed` | -0.631 |
| `depth-10` | improvements | `skating-missed-jumps` | 0.123 |
| `depth-10` | improvements | `skating-table-worst-flight-time` | 0.387 |
| `depth-10` | improvements | `skating-rotation-doubles-triples` | 0.369 |
| `expansion-none` | improvements | `eval-follow-up-ordering` | 0.369 |
| `expansion-deterministic` | regressions | `eval-what-cannot-be-generated` | -0.080 |
| `expansion-deterministic` | regressions | `eval-follow-up-ordering` | -0.369 |
| `chunk-256-25` | regressions | `primer-follow-up-generation` | -0.500 |
| `chunk-256-25` | regressions | `skating-height-underestimate` | -0.157 |
| `chunk-256-25` | regressions | `eval-follow-up-ordering` | -0.200 |
| `chunk-256-25` | regressions | `notes-provider-failure` | -0.369 |
| `chunk-256-25` | regressions | `notes-injection-not-agreed` | -0.307 |
| `chunk-256-25` | regressions | `notes-follow-up-hole` | -0.500 |
| `chunk-256-25` | improvements | `skating-errors` | 0.431 |
| `chunk-256-25` | improvements | `skating-missed-jumps` | 0.226 |
| `chunk-256-25` | improvements | `skating-table-worst-flight-time` | 0.044 |
| `chunk-256-25` | improvements | `eval-citation-pair` | 0.226 |
| `chunk-256-25` | improvements | `eval-abstention-errors` | 0.369 |
| `chunk-256-25` | improvements | `eval-what-cannot-be-generated` | 0.080 |
| `chunk-1024-100` | regressions | `skating-errors` | -0.431 |
| `chunk-1024-100` | regressions | `skating-missed-jumps` | -0.226 |
| `chunk-1024-100` | regressions | `eval-citation-pair` | -0.226 |
| `chunk-1024-100` | regressions | `eval-abstention-errors` | -0.369 |
| `chunk-1024-100` | regressions | `eval-what-cannot-be-generated` | -0.080 |
| `chunk-1024-100` | improvements | `primer-follow-up-generation` | 0.500 |
| `chunk-1024-100` | improvements | `skating-table-worst-flight-time` | 0.200 |
| `chunk-1024-100` | improvements | `skating-table-successful-pps` | 1.000 |
| `chunk-1024-100` | improvements | `eval-follow-up-ordering` | 0.200 |
| `chunk-1024-100` | improvements | `notes-provider-failure` | 0.369 |
| `chunk-1024-100` | improvements | `notes-injection-not-agreed` | 0.307 |
| `chunk-1024-100` | improvements | `notes-follow-up-hole` | 0.500 |
| `chunk-1024-100` | improvements | `skating-rotation-speed` | 0.500 |

## Reproducing this report

```
docker compose up db qdrant
uv run python -m evaluation.cli --live --report <directory> --ablate
uv run python -m evaluation.cli --live --report <directory> --compare-report <published directory>
```

The manifest holds the revision, the case set, the document hashes, the prompts, the models, the settings, and the environment; each result beside it holds the index manifest and generation of every document, with the per-question rows the numbers came from. A run that measured the same things reproduces the manifest; a run that did not names the field that moved, and a changed model revision is reported on its own.
