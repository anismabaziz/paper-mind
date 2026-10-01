# Retrieval baseline: case set 2026-09-labeled-cases-v1

Measured on revision `d85d2514c9ac` (fix: report how far a re-run's numbers moved, not only whether the setup matched)
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
| `baseline` | baseline | - | 0.730 | 0.716 | 0.654 | 0.612 | 0.0122 | 0.0385 |
| `dense-only` | retrieval-method | method=dense | 0.730 | 0.716 | 0.598 | 0.575 | 0.0095 | 0.0132 |
| `sparse-only` | retrieval-method | method=sparse | 0.757 | 0.757 | 0.698 | 0.663 | 0.0061 | 0.0124 |
| `hybrid-only` | retrieval-method | method=hybrid | 0.730 | 0.716 | 0.654 | 0.612 | 0.0100 | 0.0163 |
| `rerank-on` | rerank | rerank=True | 0.757 | 0.757 | 0.651 | 0.629 | 0.1726 | 0.9934 |
| `depth-10` | candidate-depth | candidate_depth=10 | 0.730 | 0.716 | 0.641 | 0.608 | 0.0138 | 0.0324 |
| `depth-100` | candidate-depth | candidate_depth=100 | 0.730 | 0.716 | 0.641 | 0.606 | 0.0143 | 0.0290 |
| `expansion-none` | query-expansion | query_expansion=none | 0.730 | 0.716 | 0.654 | 0.616 | 0.0125 | 0.0377 |
| `expansion-deterministic` | query-expansion | query_expansion=deterministic | 0.730 | 0.716 | 0.654 | 0.612 | 0.0118 | 0.0224 |
| `chunk-256-25` | chunking | chunk_size_tokens=256, chunk_overlap_tokens=25 | 0.784 | 0.784 | 0.625 | 0.615 | 0.0421 | 0.0624 |
| `chunk-1024-100` | chunking | chunk_size_tokens=1024, chunk_overlap_tokens=100 | 0.784 | 0.784 | 0.658 | 0.648 | 0.0094 | 0.0155 |

## Questions that moved

| Experiment | Direction | Question | nDCG change |
| --- | --- | --- | --- |
| `dense-only` | regressions | `skating-missed-jumps` | -0.080 |
| `dense-only` | regressions | `skating-conclusion` | -0.569 |
| `dense-only` | regressions | `eval-correctness` | -0.369 |
| `dense-only` | regressions | `eval-citation-pair` | -0.226 |
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
| `hybrid-only` | regressions | `skating-height-underestimate` | -0.080 |
| `hybrid-only` | regressions | `eval-abstention-errors` | -0.369 |
| `hybrid-only` | regressions | `eval-what-cannot-be-generated` | -0.080 |
| `hybrid-only` | regressions | `skating-rotation-speed` | -1.000 |
| `hybrid-only` | improvements | `skating-false-positive` | 0.500 |
| `hybrid-only` | improvements | `skating-table-worst-flight-time` | 0.387 |
| `rerank-on` | regressions | `skating-missed-jumps` | -0.123 |
| `rerank-on` | regressions | `skating-table-worst-flight-time` | -0.387 |
| `rerank-on` | regressions | `skating-rotation-doubles-triples` | -0.369 |
| `rerank-on` | improvements | `skating-errors` | 0.500 |
| `rerank-on` | improvements | `skating-height-underestimate` | 0.387 |
| `rerank-on` | improvements | `skating-rotation-speed` | 0.631 |
| `depth-10` | regressions | `skating-errors` | -0.500 |
| `depth-10` | regressions | `skating-height-underestimate` | -0.387 |
| `depth-10` | regressions | `eval-citation-pair` | -0.226 |
| `depth-10` | regressions | `skating-rotation-speed` | -0.631 |
| `depth-10` | improvements | `skating-missed-jumps` | 0.123 |
| `depth-10` | improvements | `skating-table-worst-flight-time` | 0.387 |
| `depth-10` | improvements | `skating-rotation-doubles-triples` | 0.369 |
| `depth-10` | improvements | `eval-what-cannot-be-generated` | 0.080 |
| `depth-100` | regressions | `eval-what-cannot-be-generated` | -0.080 |
| `expansion-none` | improvements | `eval-follow-up-ordering` | 0.369 |
| `expansion-deterministic` | regressions | `eval-follow-up-ordering` | -0.369 |
| `expansion-deterministic` | improvements | `eval-citation-pair` | 0.226 |
| `chunk-256-25` | regressions | `primer-follow-up-generation` | -0.500 |
| `chunk-256-25` | regressions | `eval-follow-up-ordering` | -0.200 |
| `chunk-256-25` | regressions | `notes-injection-not-agreed` | -0.307 |
| `chunk-256-25` | regressions | `notes-follow-up-hole` | -0.500 |
| `chunk-256-25` | improvements | `skating-errors` | 0.431 |
| `chunk-256-25` | improvements | `skating-rotation-threshold` | 0.387 |
| `chunk-256-25` | improvements | `skating-missed-jumps` | 0.226 |
| `chunk-256-25` | improvements | `skating-table-worst-flight-time` | 0.044 |
| `chunk-256-25` | improvements | `skating-height-underestimate` | 0.080 |
| `chunk-256-25` | improvements | `eval-abstention-errors` | 0.369 |
| `chunk-256-25` | improvements | `eval-what-cannot-be-generated` | 0.080 |
| `chunk-1024-100` | regressions | `skating-errors` | -0.431 |
| `chunk-1024-100` | regressions | `skating-rotation-threshold` | -0.387 |
| `chunk-1024-100` | regressions | `skating-missed-jumps` | -0.226 |
| `chunk-1024-100` | regressions | `eval-citation-pair` | -0.226 |
| `chunk-1024-100` | regressions | `eval-abstention-errors` | -0.369 |
| `chunk-1024-100` | improvements | `primer-follow-up-generation` | 0.500 |
| `chunk-1024-100` | improvements | `skating-table-worst-flight-time` | 0.200 |
| `chunk-1024-100` | improvements | `skating-table-successful-pps` | 0.631 |
| `chunk-1024-100` | improvements | `eval-follow-up-ordering` | 0.200 |
| `chunk-1024-100` | improvements | `notes-injection-not-agreed` | 0.307 |
| `chunk-1024-100` | improvements | `notes-follow-up-hole` | 0.500 |
| `chunk-1024-100` | improvements | `skating-rotation-speed` | 0.500 |

## Reproducing this report

```
docker compose up db qdrant
uv run python -m evaluation.cli --live --report <directory> --ablate
uv run python -m evaluation.cli --live --report <directory> --compare-report <published directory>
```

The manifest holds the revision, the case set, the document hashes, the prompts, the models, the settings, and the environment; each result beside it holds the index manifest and generation of every document, with the per-question rows the numbers came from. A re-run says whether the setup matched and how far each experiment's numbers moved, and a changed model revision is named on its own.

The setup is checkable; the numbers are a sample. Retrieval is measured against an approximate index whose sparse scores carry term weights taken across the whole collection, so two runs over the same documents in the same collection can rank a borderline passage differently. Treat a difference smaller than the reported movement as no difference.
