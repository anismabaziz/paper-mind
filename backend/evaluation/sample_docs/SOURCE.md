# Sample documents

These documents exist so the evaluation set has something real to score
against and so the demo has documents worth uploading. Licenses:

- `papermind-rag-primer.pdf` — authored in-repo for this project; the
  markdown source sits next to this file. License: CC0 1.0 (public
  domain).
- `papermind-eval-methods.pdf` — authored in-repo for this project; the
  markdown source sits next to this file. License: CC0 1.0 (public
  domain). It describes how the evaluation is measured, and its table of
  configurations is measured over the case set's fourteen tuning cases.
- `papermind-team-notes.pdf` — authored in-repo for this project; the
  markdown source sits next to this file. License: CC0 1.0 (public
  domain). It carries a pasted notice written to look like an
  instruction, left in on purpose so the case set has a document to ask
  about when checking that document text cannot change how a reader is
  answered.
- `bruening-2018-wearable-jump-monitor-figure-skating.pdf` — Bruening,
  Reynolds, Adair, Zapalo & Ridge (2018), "A sport-specific wearable jump
  monitor for figure skating", PLOS ONE,
  doi:10.1371/journal.pone.0206162. License: CC BY 4.0, unchanged from the
  publisher's version apart from the filename.

The three written here are rebuilt together with
`python -m evaluation.build_sample_pdfs` after editing their markdown. The
build is reproducible, so an unchanged source produces the same bytes, which
is what lets the case set pin each document by content hash. The paper is
never regenerated.

`../datasets/` pairs each document with the questions asked about it, the
passages that answer them, and the outcome each question has to reach. The same
documents drive the retrieval evaluator and are the intended upload set for the
live demo.
