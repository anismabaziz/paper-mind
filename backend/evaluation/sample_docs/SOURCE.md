# Sample documents

These documents exist so the evaluation fixture has something real to
score against and so the demo has documents worth uploading. Licenses:

- `papermind-rag-primer.pdf` — authored in-repo for this project; the
  markdown source sits next to this file. License: CC0 1.0 (public
  domain). Regenerate the PDF with
  `python -m evaluation.build_primer_pdf` after editing the source.
- `bruening-2018-wearable-jump-monitor-figure-skating.pdf` — Bruening,
  Reynolds, Adair, Zapalo & Ridge (2018), "A sport-specific wearable jump
  monitor for figure skating", PLOS ONE,
  doi:10.1371/journal.pone.0206162. License: CC BY 4.0, unchanged from the
  publisher's version apart from the filename.

`../fixture.json` pairs each document with questions, expected answers,
and gold snippets. The same fixture drives the retrieval evaluator and is
the intended upload set for the live demo.
