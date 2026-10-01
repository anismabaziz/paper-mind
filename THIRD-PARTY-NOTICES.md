# Third-party notices

PaperMind's own code is under the [MIT License](../LICENSE). The resources
below ship with the repository or are fetched automatically at install or
first run, and each keeps its upstream license.

## Sample Documents (`backend/evaluation/sample_docs/`)

Three fixtures were written for this project and are public domain under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). Their Markdown
sources sit next to the PDFs and rebuild byte-identically with
`python -m evaluation.build_sample_pdfs`:

- `papermind-rag-primer.pdf`
- `papermind-eval-methods.pdf`
- `papermind-team-notes.pdf` — carries a pasted notice written to look like
  an instruction, left in on purpose so the evaluation set can check that
  Document text cannot change how a reader is answered.

One fixture is someone else's paper, unchanged apart from the filename:

- `bruening-2018-wearable-jump-monitor-figure-skating.pdf` — Bruening,
  Reynolds, Adair, Zapalo & Ridge (2018), "A sport-specific wearable jump
  monitor for figure skating", PLOS ONE,
  doi:[10.1371/journal.pone.0206162](https://doi.org/10.1371/journal.pone.0206162),
  under [CC BY 4.0 / CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

The `--seed` demo indexes the two CC0 primers only, so a fresh clone needs
no rights beyond the public domain.

## PDF reader assets (`frontend/public/cmaps/`, `frontend/public/standard_fonts/`)

Vendored from [PDF.js](https://github.com/mozilla/pdf.js) (`pdfjs-dist` npm package, Mozilla,
[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)) by
`frontend/scripts/vendor-pdf-assets.mjs`. The character maps and the
standard fonts inherit their own terms:

- Foxit fonts (`Foxit*.pfb`) — BSD-style license from the PDFium authors;
  see `frontend/public/standard_fonts/LICENSE_FOXIT`.
- Liberation fonts (`LiberationSans-*.ttf`) — digitized data under the
  [SIL Open Font License 1.1](https://scripts.sil.org/OFL); see
  `frontend/public/standard_fonts/LICENSE_LIBERATION`.

## Local model weights (downloaded at first run, not bundled)

The embedding model and the rerankers are fetched from Hugging Face into the
local cache (`HF_HOME`) at the revisions pinned in
`backend/services/models.py`, under their upstream licenses:

- `BAAI/bge-m3` (MIT) — dense and sparse-capable encoder.
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (Apache-2.0) — default gated reranker.
- `BAAI/bge-reranker-v2-m3` (MIT) — heavier reranker an operator can select.

## Dependencies (resolved, not bundled)

Backend packages resolve from `backend/uv.lock` (`uv sync --frozen`) and
frontend packages from `frontend/package-lock.json` (`npm ci`). Those trees
stay under their own licenses; notable copyleft-free core pieces are Flask,
SQLAlchemy, React, Vite, pdfjs-dist, and Postgres/Qdrant server images, which
are pinned by digest in `backend/compose.yaml` rather than shipped.
