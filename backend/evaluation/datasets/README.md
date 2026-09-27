# The labeled case set

Each version is a directory holding one `dataset.json`, which pairs a set of
reviewed questions with the passages that answer them and what the run has to do
with each. `CURRENT_DATASET` in `../dataset.py` names the version a run measures
by default. Nothing is ever edited in place. A change to a case makes a new version
directory, `CHANGELOG.md` records what moved, and the version a report carries
says which set produced it.

## What a case holds

```json
{
  "id": "skating-candidate-counts",
  "document": "bruening-2018-wearable-jump-monitor-figure-skating.pdf",
  "split": "tuning",
  "category": "numeric_reasoning",
  "question": "How many candidate peak pairs did the algorithm start from, ...",
  "expected_outcome": "answered",
  "expected_answer": "5,807 peak pairs were candidates; the count was reduced ...",
  "expected_evidence": ["A total of 5,807 peak pairs were present, ..."],
  "rubric": "A correct answer gives 5,807 candidate peak pairs and says ..."
}
```

- `split` is `tuning` or `validation`. The tuning split is read while a
  configuration is being chosen, and the reported split is read once and quoted.
  The split is made before any configuration is picked, and it does not move.
- `category` is how the case is asked, and the set is balanced over it: exact
  lookup, paraphrase, numeric, table, multi-section evidence, unanswerable,
  follow-up, prompt injection, citation validation, provider failure.
- `expected_outcome` is the outcome the run has to end as. It is declared rather
  than inferred, so a grader is never handed an expectation the set did not
  write down.
- `expected_evidence` are passages from the document, quoted closely enough that
  a citation to the right sentence can be recognised. A question that restates
  the passage it expects is refused.
- `rubric` says what a person checking the answer should find, and it reaches the
  judge. A case a reviewer cannot reproduce is not a reviewed case.
- `absent_terms` are the words whose presence in the document would make an
  unanswerable question answerable. They are checked against the document, so an
  unanswerable case cannot quietly become answerable.
- `fault` marks a case that only means anything in a run that injects a failure.
  It declares which failure, and the outcome that failure has to come out as.
  Those cases are held back from a run and named in the run record, so a
  reported number is never a count of deliberately broken providers.
- `notes` is for the reviewer. It never reaches a report.

## Adding or changing a case

1. Write the question, then find the passage that answers it and quote that into
   `expected_evidence`. Run the tests: the passage has to be in the document, on
   one page, and not restated in the question.
2. Write the expected answer, then the rubric saying what a correct answer has to
   contain. If the rubric cannot be written, the case is not ready.
3. For a question the document cannot answer, name the terms that would answer
   it and leave the evidence out. If the document holds one of those terms, the
   question is not unanswerable yet.
4. Adding or removing cases changes what the numbers mean. Make a new version
   directory, point `CURRENT_DATASET` at it, and write the change down in
   `CHANGELOG.md`.

## The documents

A document entry pins the exact bytes its cases were reviewed against, with its
version, where it came from, and its license. Replacing or regenerating a
document fails validation until someone re-reads the cases against the new bytes
and updates the pin. The three documents written for this project are rebuilt
with `python -m evaluation.build_sample_pdfs`, which produces the same bytes for
the same markdown; the published paper is never regenerated.
