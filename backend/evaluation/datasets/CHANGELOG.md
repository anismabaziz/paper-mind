# Case set changelog

Every version of the labeled case set, what changed in it, and what a report
measured against an earlier one can still be compared with. A run records the
version it measured, so a number is only meaningful next to the set that
produced it.

## 2026-09-labeled-cases-v1

Reviewed 2026-09-27. 57 cases over four documents: 17 in the tuning split, 40
in the reported one.

Documents added:

- `papermind-eval-methods.pdf`, authored in-repo under CC0. It carries the
  metric definitions, a measured comparison table, the split rule, and the
  limits of a set this size.
- `papermind-team-notes.pdf`, authored in-repo under CC0. It carries the
  reading rules for a run report and a pasted notice shaped like an
  instruction, so the set has a document to ask about for the injection cases.

Documents changed:

- `papermind-rag-primer.pdf` is at version 2. The text is unchanged; the file
  was rebuilt so that building it twice produces the same bytes, which is what
  lets the content pin in the set mean something.

Cases changed:

- The ten cases carried over from the previous set keep their ids, questions,
  and expected answers. One of them had evidence that appeared on more than one
  page, which made a page citation ambiguous, so `skating-sensor-placement` now
  points at the sentence saying where the monitor was fitted.
- The unanswerable cases expect an answer that declines rather than an
  abstention. Retrieval fills its budget from a document that has something to
  fill it with, so the app abstains only when it retrieved nothing at all, and a
  case that demanded an abstention would have reported correct behaviour as a
  defect. Either is now accepted, so the abstention grader reports nothing for
  them and the answer itself is graded instead.
