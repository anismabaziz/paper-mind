# Research Brief tool use: task set 2026-09-brief-tasks-v1

7 tasks over 24 deterministic trials. Each task is measured several times because a brief chooses its own tool order; consistency across trials is reported beside quality so one lucky run cannot pass the set.

## What this report measured

- **outcome**: evidence coverage, correctness, citation precision and recall, page citations, conflict handling, gaps, and abstention, graded deterministically from recorded trajectories
- **trajectory**: tool-call validity, scope enforcement, repeated calls, unnecessary calls, and completion status for every trial
- **equivalence**: coverage is a set comparison over held evidence, so two different orders of the same searches grade the same; no task requires one exact tool path
- **paid model trials**: not measured here — they run on a controlled schedule with a cost limit ($5.00 per run, 3 trials per task)

## Totals

Trials 24, latency p50 12.25s p95 23.70s, turns p50 3.0 p95 3.0, tool calls p50 3.5 p95 4.0, tokens 28885 (p50 1237.5 p95 1377.8), cost $0.1860, timeout rate 4%.
Failures: timeout 1.

## Tasks

| Task | Category | Trials | Consistency | Coverage | Correctness | Conflict | Abstention | p50 / p95 (s) | Cost ($) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `brief-cross-lookup-hit-rate` | cross_lookup | 5 | 80% | 60% | 80% | 100% | 80% | 6.00s / 25.30s | 0.0300 |
| `brief-agreement-recall-stricter` | agreement | 3 | 100% | 100% | 100% | 100% | 100% | 8.00s / 8.90s | 0.0240 |
| `brief-disagreement-single-representation` | disagreement | 3 | 100% | 100% | 100% | 100% | 100% | 10.00s / 11.80s | 0.0240 |
| `brief-missing-evidence-model` | missing_evidence | 4 | 100% | 100% | 100% | 100% | 100% | 12.00s / 14.70s | 0.0360 |
| `brief-unanswerable-hardware-cost` | unanswerable | 3 | 100% | - | - | - | 100% | 16.00s / 17.80s | 0.0240 |
| `brief-page-citation-rerank-table` | page_citation | 3 | 100% | 100% | 100% | 100% | 100% | 19.00s / 20.80s | 0.0240 |
| `brief-compare-rerank-worth` | agreement | 3 | 100% | 100% | 100% | 100% | 100% | 22.00s / 23.80s | 0.0240 |

## Configurations

Direct retrieval asks once with no tools; the brief searches and reads; the extended brief may also read pages and compare evidence.

| Configuration | Trials | Coverage | Correctness | Citation | Conflict | Abstention | p50 / p95 (s) | Cost ($) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `direct` | 1 | 0% | 100% | - | 100% | 100% | 1.20s / 1.20s | 0.0020 |
| `brief` | 11 | 88% | 88% | 100% | 100% | 91% | 12.50s / 24.00s | 0.0880 |
| `extended` | 12 | 100% | 100% | 100% | 100% | 100% | 13.00s / 22.90s | 0.0960 |

## Representative trajectories

### Successful: `brief-cross-lookup-hit-rate` trial 1 (brief, complete, 3 turns, 3 tool calls, 4.00s, $0.0080)

Tools: search_passages, search_passages, read_passages.

- supported: The share of questions for which at least one gold passage appears in the top k results; the mean over the set is the share of questions the system can reach at all. (supports E1, E2; conflicts —)

### Partial: `brief-missing-evidence-model` trial 10 (brief, partial, 3 turns, 3 tool calls, 13.00s, $0.0080)

Tools: search_passages, search_passages, read_passages.

- supported: The primer names no embedding model, so that half cannot be answered from the pair; it does say overlap between consecutive chunks reduces the chance that a fact sitting across a cut boundary is lost  (supports E1; conflicts —)
- gap: The pair names no embedding model, so that half is unanswered.

### Conflicting: `brief-disagreement-single-representation` trial 7 (extended, complete, 3 turns, 4 tool calls, 10.00s, $0.0080)

Tools: search_passages, search_passages, read_passages, compare_evidence.

- contested: Sparse finds more of the gold passages (recall 0.70 against 0.62) while dense reaches more questions outright (hit rate 0.71 against 0.66); neither wins on both, so the finding is contested between th (supports E1; conflicts E2)

### Failed: `brief-cross-lookup-hit-rate` trial 24 (brief, timeout, 2 turns, 1 tool calls, 30.00s, $0.0040)

Tools: search_passages.


## Reproducing this report

```
uv run pytest backend/tests/test_brief_evaluation.py
uv run python -m evaluation.cli --brief-eval --brief-report <directory>
```

Deterministic checks run routinely with no key. Paid model trials that record new trajectories run on a controlled schedule: `--brief-eval --allow-paid` refuses when the estimate is past the $5.00 limit.
