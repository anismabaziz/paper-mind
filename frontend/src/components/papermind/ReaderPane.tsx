import { lazy, Suspense, useRef, useState, useEffect } from "react";
import {
  ChevronLeft,
  ChevronRight,
  Highlighter,
  List,
  Minus,
  Plus,
  Quote,
  StickyNote,
  UploadIcon,
  Trash2,
  MoreHorizontal,
  Columns2,
} from "lucide-react";
import usePdfStore from "@/store/pdf-state";
import { checkIsProcessed, deleteFile } from "@/services/files";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { cn } from "@/lib/utils";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu";

const ReaderDocument = lazy(() => import("./ReaderDocument"));

const outline = [
  { id: "sec-intro", label: "I. Introduction", page: 1 },
  { id: "sec-why", label: "II. Why Access Control Is Not Enough", page: 2 },
  { id: "sec-drift", label: "III. Codified Policies and Reasoning Drift", page: 3 },
];

export function ReaderPane() {
  const { file } = usePdfStore();
  const queryClient = useQueryClient();
  const scrollRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(100);
  const [showOutline, setShowOutline] = useState(true);
  const [progress, setProgress] = useState(18);
  const [page, setPage] = useState(1);
  const [numPages, setNumPages] = useState<number | null>(null);

  const checkProcessedQuery = useQuery({
    queryKey: [file?.name, "is-processed"],
    queryFn: () => checkIsProcessed(file!),
    enabled: !!file,
  });

  const deleteMutation = useMutation({
    mutationFn: deleteFile,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["files"] }),
  });

  const isProcessed = checkProcessedQuery.data?.is_processed ?? false;

  useEffect(() => {
    // reset page when file changes
    setPage(1);
    setNumPages(null);
  }, [file?.id]);

  useEffect(() => {
    if (numPages && page > numPages) setPage(numPages);
  }, [numPages, page]);

  function onScroll() {
    const el = scrollRef.current;
    if (!el) return;
    const ratio = el.scrollTop / Math.max(1, el.scrollHeight - el.clientHeight);
    setProgress(Math.round(18 + ratio * 60));
  }

  return (
    <main className="flex min-w-0 flex-1 flex-col bg-canvas">
      {/* Toolbar */}
      <header className="flex h-14 items-center justify-between gap-4 border-b border-rule bg-background/80 px-5 backdrop-blur">
        <div className="flex min-w-0 items-center gap-3">
          <button
            type="button"
            onClick={() => setShowOutline((v) => !v)}
            className={cn(
              "flex size-7 items-center justify-center rounded-sm border transition-colors",
              showOutline ? "border-ink bg-ink text-paper" : "border-rule text-ink-soft hover:border-ink",
            )}
            aria-label="Toggle outline"
          >
            <List className="size-3.5" />
          </button>
          <div className="min-w-0">
            <p className="truncate font-serif text-[0.95rem] leading-tight">
              {file ? file.name.replace(/\.[^/.]+$/, "") : "Document Viewer"}
            </p>
            <p className="label-meta truncate">
              {file ? `${file.metadata.content_type} · ${isProcessed ? "Indexed" : "Indexing"}` : "No document selected"}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="hidden items-center gap-1.5 lg:flex">
            <button
              type="button"
              className="flex items-center gap-1.5 rounded-sm border border-rule px-2 py-1 text-[0.72rem] text-ink-soft transition-colors hover:border-ink hover:text-ink"
            >
              <Highlighter className="size-3" /> Highlight
            </button>
            <button
              type="button"
              className="flex items-center gap-1.5 rounded-sm border border-rule px-2 py-1 text-[0.72rem] text-ink-soft transition-colors hover:border-ink hover:text-ink"
            >
              <StickyNote className="size-3" /> Note
            </button>
            <button
              type="button"
              className="flex items-center gap-1.5 rounded-sm border border-rule px-2 py-1 text-[0.72rem] text-ink-soft transition-colors hover:border-ink hover:text-ink"
            >
              <Quote className="size-3" /> Ask
            </button>
          </div>

          {file && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button
                  type="button"
                  className="flex size-7 items-center justify-center rounded-sm border border-rule text-ink-soft hover:border-ink hover:text-ink"
                >
                  <MoreHorizontal className="size-3.5" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="border-rule bg-paper">
                <DropdownMenuItem
                  className="text-destructive focus:bg-destructive/10 focus:text-destructive"
                  onClick={() => deleteMutation.mutate(file)}
                >
                  <Trash2 className="size-3.5" /> Delete Permanently
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          )}

          <div className="flex items-center gap-1 border-l border-rule pl-4">
            <button
              type="button"
              onClick={() => setZoom((z) => Math.max(80, z - 10))}
              className="flex size-6 items-center justify-center text-ink-soft hover:text-ink"
              aria-label="Zoom out"
            >
              <Minus className="size-3" />
            </button>
            <span className="w-10 text-center font-mono text-[0.68rem] text-ink-soft">{zoom}%</span>
            <button
              type="button"
              onClick={() => setZoom((z) => Math.min(140, z + 10))}
              className="flex size-6 items-center justify-center text-ink-soft hover:text-ink"
              aria-label="Zoom in"
            >
              <Plus className="size-3" />
            </button>
          </div>

          <div className="flex items-center gap-1 border-l border-rule pl-4">
            <button
              type="button"
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              className="flex size-6 items-center justify-center text-ink-soft hover:text-ink"
              aria-label="Previous page"
            >
              <ChevronLeft className="size-3.5" />
            </button>
            <span className="font-mono text-[0.68rem] text-ink-soft">
              {String(page).padStart(2, "0")} / {String(numPages ?? 0).padStart(2, "0")}
            </span>
            <button
              type="button"
              onClick={() => setPage((p) => Math.min(numPages ?? p, p + 1))}
              className="flex size-6 items-center justify-center text-ink-soft hover:text-ink"
              aria-label="Next page"
            >
              <ChevronRight className="size-3.5" />
            </button>
          </div>
        </div>
      </header>

      <div className="relative flex min-h-0 flex-1">
        {/* Outline + thumbnails */}
        {showOutline && (
          <div className="scroll-slim hidden w-56 shrink-0 overflow-y-auto border-r border-rule bg-background/50 px-4 py-5 xl:block">
            <p className="label-meta pb-3">Contents</p>
            <ul className="space-y-1">
              {outline.map((o) => (
                <li key={o.id}>
                  <button
                    type="button"
                    onClick={() => setPage(o.page)}
                    className={cn(
                      "flex w-full items-baseline gap-2 rounded-sm px-2 py-1.5 text-left text-[0.8rem] leading-snug transition-colors",
                      page === o.page ? "bg-marker-soft text-ink" : "text-ink-soft hover:bg-paper hover:text-ink",
                    )}
                  >
                    <span className="font-mono text-[0.62rem] text-ink-faint">{o.page}</span>
                    {o.label}
                  </button>
                </li>
              ))}
            </ul>

            <p className="label-meta pt-6 pb-3">Pages</p>
            <div className="grid grid-cols-2 gap-2">
              {numPages
                ? Array.from({ length: numPages }, (_, i) => i + 1).map((n) => (
                    <button
                      key={n}
                      type="button"
                      onClick={() => setPage(n)}
                      className={cn(
                        "group relative aspect-[3/4] overflow-hidden rounded-[2px] border bg-paper p-1.5 transition-all",
                        page === n ? "border-marker shadow-sheet" : "border-rule opacity-70 hover:opacity-100",
                      )}
                    >
                      <span className="flex h-full flex-col gap-[3px]">
                        {Array.from({ length: 11 }).map((_, i) => (
                          <span key={i} className="block h-[2px] rounded-full bg-ink/15" style={{ width: `${55 + ((i * 37) % 45)}%` }} />
                        ))}
                      </span>
                      <span className="absolute right-1 bottom-1 font-mono text-[0.55rem] text-ink-faint">{n}</span>
                    </button>
                  ))
                : Array.from({ length: 4 }, (_, i) => i + 1).map((n) => (
                    <div
                      key={n}
                      className="relative aspect-[3/4] overflow-hidden rounded-[2px] border border-rule bg-paper p-1.5 opacity-40"
                    >
                      <span className="flex h-full flex-col gap-[3px]">
                        {Array.from({ length: 11 }).map((_, i) => (
                          <span key={i} className="block h-[2px] rounded-full bg-ink/15" style={{ width: `${55 + ((i * 37) % 45)}%` }} />
                        ))}
                      </span>
                      <span className="absolute right-1 bottom-1 font-mono text-[0.55rem] text-ink-faint">{n}</span>
                    </div>
                  ))}
            </div>

            {file && (
              <div className="mt-6 border-t border-rule pt-4">
                <p className="label-meta pb-2">Document</p>
                <p className="font-serif text-xs leading-snug">{file.name}</p>
                <p className="mt-1 font-mono text-[0.62rem] text-ink-faint">{(file.metadata.size / 1024).toFixed(0)} KB · {file.metadata.content_type}</p>
                <span
                  className={cn(
                    "mt-2 inline-flex border px-2 py-0.5 font-mono text-[0.6rem] uppercase tracking-wider",
                    isProcessed ? "border-ink bg-ink text-paper" : "border-rule bg-paper text-ink-faint",
                  )}
                >
                  {isProcessed ? "Indexed" : "Indexing"}
                </span>
              </div>
            )}
          </div>
        )}

        {/* Reading sheet */}
        <div ref={scrollRef} onScroll={onScroll} className="scroll-slim flex-1 overflow-y-auto px-6 py-8">
          {!file ? (
            <div className="mx-auto flex min-h-[520px] max-w-[560px] flex-col items-center justify-center">
              <div className="paper-grain w-full bg-paper px-10 py-16 text-center shadow-sheet">
                <div className="mx-auto grid size-10 place-items-center border border-rule bg-canvas text-ink-faint">
                  <UploadIcon className="size-5" />
                </div>
                <h2 className="mt-6 font-serif text-xl font-medium">No document selected</h2>
                <p className="mx-auto mt-2 max-w-[32ch] font-serif text-sm leading-relaxed text-ink-soft">
                  Select a publication from your library to review its contents in this pane — the sheet keeps your margins clean.
                </p>
                <p className="label-meta mt-6">Ingest a PDF from the rail to begin</p>
              </div>
            </div>
          ) : (
            <div
              className="mx-auto origin-top transition-[width] duration-200"
              style={{ width: `${Math.min(880, 7.6 * zoom)}px`, maxWidth: "100%" }}
            >
              {/* Title sheet */}
              <article className="paper-grain mb-6 bg-paper px-10 pt-10 pb-8 shadow-sheet">
                <p className="label-meta">Research paper · {file.metadata.content_type}</p>
                <h1 className="mt-3 font-serif text-[1.7rem] leading-[1.15] font-medium text-balance">
                  {file.name.replace(/\.[^/.]+$/, "")}
                </h1>
                <div className="mt-5 flex items-center gap-3 border-y border-rule py-4">
                  <span
                    className={cn(
                      "inline-flex border px-2 py-1 font-mono text-[0.62rem] uppercase tracking-wider",
                      isProcessed ? "border-marker bg-marker-soft text-marker" : "border-rule bg-canvas text-ink-faint",
                    )}
                  >
                    {isProcessed ? "Ready for questions" : "Indexing… answers paused"}
                  </span>
                  <span className="font-mono text-[0.62rem] text-ink-faint">
                    Page {String(page).padStart(2, "0")} · {file.name}
                  </span>
                </div>
                <p className="mt-4 font-serif text-[0.95rem] leading-[1.7] text-ink-soft italic">
                  <span className="mr-2 font-mono text-[0.62rem] tracking-[0.14em] text-marker not-italic uppercase">Abstract</span>
                  This workspace keeps every answer tied to the passage it came from. Ask a question in the companion and the document stays open beside it — no context lost.
                </p>
              </article>

              {/* PDF sheet */}
              <div className="paper-grain relative bg-paper shadow-sheet">
                <div className="flex items-center justify-between border-b border-rule px-6 py-3">
                  <span className="label-meta">Page {String(page).padStart(2, "0")}</span>
                  <span className="h-px flex-1 mx-3 bg-rule" />
                  <span className="label-meta">{file.name} · p. {page}</span>
                </div>
                <div className="relative bg-canvas p-3">
                  <div className="overflow-hidden border border-rule bg-white">
                    <Suspense
                      fallback={
                        <div className="grid h-[760px] place-items-center bg-white">
                          <p className="font-mono text-xs text-ink-faint">Loading document…</p>
                        </div>
                      }
                    >
                      <ReaderDocument file={file} page={page} zoom={zoom} onLoadSuccess={setNumPages} />
                    </Suspense>
                  </div>
                  {!isProcessed && (
                    <div className="pointer-events-none absolute inset-3 grid place-items-center bg-paper/70 backdrop-blur-[1px]">
                      <div className="rounded-sm border border-rule bg-paper px-4 py-3 text-center shadow-sheet">
                        <p className="font-mono text-xs font-medium">Indexing document…</p>
                        <p className="mt-1 text-xs text-ink-soft">Semantic vectors are being generated — chat will unlock when this pass finishes.</p>
                      </div>
                    </div>
                  )}
                </div>
                <div className="flex items-center justify-between border-t border-rule px-10 py-4">
                  <span className="label-meta">{file.name}</span>
                  <span className="label-meta">{String(page).padStart(2, "0")}</span>
                </div>
                {/* margin rail */}
                <span className="pointer-events-none absolute inset-y-0 left-6 hidden w-px bg-rule lg:block" />
              </div>

              {/* Figure caption mock to keep editorial rhythm */}
              <figure className="mt-6 border-y border-rule bg-paper/60 px-6 py-5">
                <div className="grid aspect-[16/7] place-items-center border border-dashed border-rule bg-canvas/50">
                  <div className="flex flex-col items-center gap-2 opacity-60">
                    <Columns2 className="size-5 text-ink-faint" />
                    <span className="label-meta">Diagram</span>
                  </div>
                </div>
                <figcaption className="mt-3 flex gap-2 text-[0.78rem] leading-snug text-ink-soft">
                  <span className="font-mono text-[0.68rem] whitespace-nowrap text-marker">Fig. 1</span>
                  Workspace layout: library, reading sheet, and grounded companion stay aligned by page and citation.
                </figcaption>
              </figure>
            </div>
          )}
        </div>

        {/* Reading progress ribbon */}
        <div className="pointer-events-none absolute top-0 right-0 bottom-0 w-1 bg-transparent">
          <div className="h-full w-px bg-rule" />
          <div className="absolute top-0 left-0 w-px bg-marker transition-[height] duration-150" style={{ height: `${progress}%` }} />
        </div>
      </div>
    </main>
  );
}
