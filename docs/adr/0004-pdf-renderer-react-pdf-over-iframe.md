# PDF renderer: react-pdf thin wrapper over iframe

Chosen: `react-pdf@10` as a thin wrapper over `pdfjs-dist@6`, loaded lazily in
a single `ReaderDocument.tsx` module. The worker is set via
`pdfjs.GlobalWorkerOptions.workerSrc = new URL('pdfjs-dist/build/pdf.worker.min.mjs', import.meta.url)`
so Vite emits it as a separate cached asset. `TextLayer.css` and
`AnnotationLayer.css` are the only viewer styles imported; all chrome is
editorial-owned (`paper-grain`, `shadow-sheet`, `rule`, `marker`,
`marker-soft`, `canvas`, `paper`). The PDF is fetched as `ArrayBuffer` with
`Authorization: Bearer` (`fetch(file.url, {headers}) → Uint8Array → <Document file={{data}}>`)
so `GET /storage/<path>` stays private. `cMapUrl` and `standardFontDataUrl`
point at `unpkg.com/pdfjs-dist@${version}` with `cMapPacked:true` for
embedded fonts. `ReaderDocument` is `React.lazy` and only mounted when
`store/pdf-state` `file != null`, keeping the ~130kB gz pdfjs + ~700kB worker
off the initial page. Virtualization renders `±2` pages around the viewport
(react-pdf has no built-in virtualizer) so 100+ page papers mount ~5 canvases;
placeholders preserve scroll height and `IntersectionObserver` tracks the
active `Page`.

Rejected: keeping the `iframe` at `file.url#toolbar=0&page=N` — it cannot send
`Authorization` (blank outside `DEMO_MODE`), varies across Chrome/Firefox/Safari/iOS
(`#toolbar=0` ignored on iOS), and reloads on every `#page=` change; no
editorial control over the inner text layer. Raw `pdfjs-dist` without
`react-pdf` — same rendering outcome with more imperative boilerplate for
`Document`/`Page` lifecycle and `TextLayer` handling, no bundle win. Full
suites `react-pdf-viewer@3` / `@react-pdf/kit` — 1–1.5MB gz, locked theme
requiring viewer `default-layout.css` overrides, and `react-pdf-viewer@3`
pinned to `pdfjs 3.4.120` with `GHSA-wgrm-67xf-hhpq`. `@react-pdf/renderer` —
a PDF generator, not a viewer.

Hard to reverse: the browser `iframe` delegation, the token-in-URL temptation,
and the split between `ReaderPane` (editorial shell, outline, progress, zoom,
citation jump) and `ReaderDocument` (pdfjs engine) define the seams for
`GET /files/:name/meta`, `ISource{page}`, and thumbnail/progress handling.
Switching back would re-break authenticated loads and re-couple the product to
browser PDF plugin quirks on iOS.

Surprising: `ArrayBuffer` → `Uint8Array` → `Document file={{data}}` is
required; assigning `file.url` to `iframe.src` or `<Document file={url}>`
cannot carry the `Bearer` header. `pdfjs-dist` detaches the `ArrayBuffer` when
transferring to the worker, so the bytes must be cloned (`data.slice()`) for
watched queries and `StrictMode` double-mounts. `cMapUrl`/`standardFontDataUrl`
are not optional — without them embedded fonts and CJK glyphs render blank.
`vite.config` needs `optimizeDeps.exclude: ['pdfjs-dist']` only if dev warns
`Setting up fake worker`.

Trade-off: we own styling and auth at the cost of owning virtualization
(`±2` window + placeholder height) and scroll sync (`IntersectionObserver`
+ `scrollToPage`). The worker is a separate network fetch but cached; lazy
loading avoids paying it on the library view. Text selection and copy are
native `TextLayer` spans with selection overridden to `var(--marker-soft)`,
so Level 1 citation jumps are page-level (`flash-cite`/`mark-cited` overlay);
pixel-perfect `bbox` rect highlights are deferred to Level 2.
