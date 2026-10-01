# PaperMind frontend

React + TypeScript + Vite interface for the PaperMind workspace: library,
reader with Page strip, streaming chat with Citation Sources, Settings
dialog, and Research Brief. It talks to the Flask API (`VITE_API_URL`) and
holds no Document text of its own beyond what the reader renders.

Start it through the one-command setup in the root
[README.md](../README.md) (`./papermind.sh up --seed`), or alone with
`npm install` and `npm run dev` against a running backend. Environment
variables are documented in [.env.example](.env.example).

PDF rendering assets under `public/cmaps/` and `public/standard_fonts/` are
vendored from `pdfjs-dist` by `scripts/vendor-pdf-assets.mjs` (runs before
`dev` and `build`); their licenses are listed in the root
[THIRD-PARTY-NOTICES.md](../THIRD-PARTY-NOTICES.md).

Checks: `npm run lint`, `npm test -- --run`, `npm run build` (type-checks
first). Browser workflows live in `e2e/` and need the backing services:
`docker compose -f ../backend/compose.test.yaml up -d --wait`, then
`npm run test:e2e`.
