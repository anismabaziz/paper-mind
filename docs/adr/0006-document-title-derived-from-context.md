# Document title derived from file context

Chosen: each `FileRecord` keeps `title` (display name), `original_filename`
(raw upload name), and `filename` (stable uuid hex on disk and in Qdrant
`pdf_name`). `title` is derived by pure helper `derive_title(pdf_bytes,
original_filename)` in priority: (1) PDF metadata `Title` via `pymupdf`
trimmed and non-empty, (2) `original_filename` without extension, (3) first
non-empty heading from the parsed chunk stream. Upload writes bytes under the
uuid name, inserts the row with both title fields, and returns `title` +
`original_filename` in `POST /upload` and `GET /files`. Listing backfills
legacy rows where `title` is null or hex-like, persisting the derived value
without re-indexing vectors. Frontend displays `file.title ?? file.name` in
LibraryRail, Reader toolbar, title sheet, and metadata block, and searches
against `title`.

Rejected: random hex display — `uuid.hex` preserves stable storage/Qdrant
paths but is not human-readable. Renaming files on disk or in Qdrant to match
titles — would invalidate existing vectors and break deletion by filename.
LLM-generated titles or user-editable overrides — out of scope and adds a
write path with no reliable signal for this change. Original filename alone
— loses the PDF's own Title metadata which is often more accurate than the
drop name.

Hard to reverse: the HTTP payload now exposes `title` + `original_filename`
alongside `name`/`url`/`metadata`; Qdrant `pdf_name` stays uuid-anchored. Any
alternative title source would need to keep the same three-column invariant
and the lazy backfill on read.

Trade-off: one extra Text + VARCHAR column and a small pure helper keeps
vectors stable and requires no re-upload, at the cost of no editable titles
yet. The heuristic cascade is deterministic and cheap (no LLM call).
