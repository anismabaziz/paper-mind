import { lazy, Suspense, useRef, useState, useEffect, useMemo, useCallback } from "react";
import {
  ChevronLeft,
  ChevronRight,
  List,
  Minus,
  Plus,
  UploadIcon,
  Trash2,
  MoreHorizontal,
  PanelLeft,
  MessageSquare,
} from "lucide-react";
import { Document, Page, pdfjs } from "react-pdf";
import usePdfStore from "@/store/pdf-state";
import useMobileUi from "@/store/mobile-ui";
import { checkIsProcessed, deleteFile, getFileMeta } from "@/services/files";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { cn } from "@/lib/utils";
import { displayTitle } from "@/types/db";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu";

// Ensure worker is configured even before lazy ReaderDocument loads (for thumbnails)
if (!pdfjs.GlobalWorkerOptions.workerSrc) {
  pdfjs.GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).toString();
}

const thumbnailOptions = {
  cMapUrl: `https://unpkg.com/pdfjs-dist@${pdfjs.version}/cmaps/`,
  cMapPacked: true,
  standardFontDataUrl: `https://unpkg.com/pdfjs-dist@${pdfjs.version}/standard_fonts/`,
};

// Page strip sizing: fixed strip height, horizontal scroll with snap
const STRIP_THUMB_WIDTH = 84;
const CLAMP = "box-border max-w-full min-w-0 overflow-hidden";

const ReaderDocument = lazy(() => import("./ReaderDocument"));

function FakePageBars() {
  return (
    <span className="flex h-full flex-col gap-[3px]">
      {Array.from({ length: 11 }).map((_, i) => (
        <span key={i} className="block h-[2px] rounded-full bg-ink/15" style={{ width: `${55 + ((i * 37) % 45)}%` }} />
      ))}
    </span>
  );
}

function ThumbnailPlaceholder({ count }: { count: number }) {
  return (
    <div className="flex gap-2 overflow-hidden">
      {Array.from({ length: count }, (_, i) => i + 1).map((n) => (
        <div key={n} className={`relative h-28 shrink-0 aspect-[3/4] rounded-[2px] border border-rule bg-paper p-1.5 opacity-40 ${CLAMP}`}>
          <FakePageBars />
          <span className="absolute right-1 bottom-1 font-mono text-[0.55rem] text-ink-faint">{n}</span>
        </div>
      ))}
    </div>
  );
}

function usePdfFileData(file: { url: string } | null) {
  const [data, setData] = useState<Uint8Array | null>(null);

  useEffect(() => {
    if (!file) {
      setData(null);
      return;
    }
    const url = file.url;
    let cancelled = false;
    const controller = new AbortController();
    async function load() {
      setData(null);
      try {
        const res = await fetch(url, { signal: controller.signal });
        if (!res.ok) throw new Error(`Failed to load PDF (${res.status})`);
        const buf = await res.arrayBuffer();
        if (!cancelled) setData(new Uint8Array(buf));
      } catch {
        if (!cancelled) setData(null);
      }
    }
    load();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [file?.url]);

  return data;
}

export function ReaderPane() {
  const { file, citationTarget } = usePdfStore();
  const queryClient = useQueryClient();
  const scrollRef = useRef<HTMLDivElement>(null);
  const stripRef = useRef<HTMLDivElement>(null);
  const outlineStripRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(100);
  const [showOutline, setShowOutline] = useState(true);
  const [progress, setProgress] = useState(0);
  const [page, setPage] = useState(1);
  const [numPages, setNumPages] = useState<number | null>(null);
  const [flashedPage, setFlashedPage] = useState<number | null>(null);
  const flashTimeoutRef = useRef<number | null>(null);
  const fileData = usePdfFileData(file);

  const thumbnailFileData = useMemo(() => (fileData ? { data: fileData.slice() } : null), [fileData]);
  const isProgrammaticRef = useRef(false);
  const programmaticTimeoutRef = useRef<number | null>(null);

  const checkProcessedQuery = useQuery({
    queryKey: [file?.name, "is-processed"],
    queryFn: () => checkIsProcessed(file!),
    enabled: !!file,
  });

  const metaQuery = useQuery({
    queryKey: [file?.name, "meta"],
    queryFn: () => getFileMeta(file!.name),
    enabled: !!file,
  });

  const deleteMutation = useMutation({
    mutationFn: deleteFile,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["files"] }),
  });

  const isProcessed = checkProcessedQuery.data?.is_processed ?? false;
  const outline = metaQuery.data?.outline ?? [];
  const { setLibraryOpen, setChatOpen } = useMobileUi();

  const scrollToPage = useCallback((pageNum: number) => {
    const container = scrollRef.current;
    if (!container) return;
    const target = container.querySelector<HTMLElement>(`[data-page="${pageNum}"]`);
    if (!target) return;
    isProgrammaticRef.current = true;
    if (programmaticTimeoutRef.current) window.clearTimeout(programmaticTimeoutRef.current);
    // Compute offset relative to scroll container (includes title sheet height)
    const containerRect = container.getBoundingClientRect();
    const targetRect = target.getBoundingClientRect();
    const top = targetRect.top - containerRect.top + container.scrollTop;
    container.scrollTo({ top, behavior: "smooth" });
    programmaticTimeoutRef.current = window.setTimeout(() => {
      isProgrammaticRef.current = false;
    }, 800);
  }, []);

  useEffect(() => {
    // reset page when file changes
    setPage(1);
    setNumPages(null);
    setProgress(0);
    if (stripRef.current) stripRef.current.scrollLeft = 0;
    if (outlineStripRef.current) outlineStripRef.current.scrollLeft = 0;
  }, [file?.id]);

  useEffect(() => {
    if (stripRef.current) stripRef.current.scrollLeft = 0;
  }, [numPages]);

  useEffect(() => {
    if (numPages && page > numPages) setPage(numPages);
  }, [numPages, page]);

  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const ratio = el.scrollTop / Math.max(1, el.scrollHeight - el.clientHeight);
    setProgress(Math.round(ratio * 100));
  }, []);

  // IntersectionObserver drives active page highlight; progress ribbon follows real scroll
  useEffect(() => {
    const root = scrollRef.current;
    if (!root || numPages == null || numPages === 0) return;

    let observer: IntersectionObserver | null = null;
    let timeoutId: number | null = null;

    const setup = () => {
      const els = root.querySelectorAll<HTMLElement>("[data-page]");
      if (els.length === 0) {
        timeoutId = window.setTimeout(setup, 150);
        return;
      }
      observer = new IntersectionObserver(
        (entries) => {
          if (isProgrammaticRef.current) return;
          let best: { page: number; ratio: number } | null = null;
          for (const entry of entries) {
            if (!entry.isIntersecting) continue;
            const pg = Number((entry.target as HTMLElement).dataset.page);
            if (!Number.isFinite(pg)) continue;
            if (!best || entry.intersectionRatio > best.ratio) best = { page: pg, ratio: entry.intersectionRatio };
          }
          if (best) {
            setPage(best.page);
          }
        },
        { root, threshold: [0, 0.3, 0.5, 0.7, 1], rootMargin: "0px 0px -20% 0px" },
      );
      els.forEach((el) => observer!.observe(el));
    };

    timeoutId = window.setTimeout(setup, 50);
    return () => {
      if (timeoutId != null) window.clearTimeout(timeoutId);
      observer?.disconnect();
    };
  }, [numPages, file?.id, zoom, fileData]);

  // Citation page-jump: ChatPane sets citationTarget → scroll Page N into view
  useEffect(() => {
    if (citationTarget == null || citationTarget.page == null) return;
    const target = citationTarget.page;
    if (!Number.isFinite(target) || target < 1) return;
    if (numPages != null && target > numPages) return;
    setPage(target);
    setFlashedPage(target);
    if (flashTimeoutRef.current) window.clearTimeout(flashTimeoutRef.current);
    flashTimeoutRef.current = window.setTimeout(() => setFlashedPage(null), 1700);
    requestAnimationFrame(() => scrollToPage(target));
  }, [citationTarget, numPages, scrollToPage]);

  useEffect(() => {
    return () => {
      if (programmaticTimeoutRef.current) window.clearTimeout(programmaticTimeoutRef.current);
      if (flashTimeoutRef.current) window.clearTimeout(flashTimeoutRef.current);
    };
  }, []);

  return (
    <main className="flex min-h-0 min-w-0 flex-1 flex-col bg-canvas">
      {/* Toolbar */}
      <header className="flex h-14 items-center justify-between gap-2 sm:gap-4 border-b border-rule bg-background/80 px-3 sm:px-5 backdrop-blur">
        <div className="flex min-w-0 items-center gap-3">
          <button
            type="button"
            onClick={() => setLibraryOpen(true)}
            className="flex size-7 items-center justify-center rounded-sm border border-rule text-ink-soft hover:border-ink hover:text-ink lg:hidden"
            aria-label="Open library"
          >
            <PanelLeft className="size-3.5" />
          </button>
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
              {file ? displayTitle(file) : "Document Viewer"}
            </p>
            <p className="label-meta truncate">
              {file ? `${file.metadata.content_type} · ${isProcessed ? "Indexed" : "Indexing"}` : "No document selected"}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 sm:gap-4">
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

          <div className="flex items-center gap-1 border-l border-rule pl-2 sm:pl-4">
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

          <div className="flex items-center gap-1 border-l border-rule pl-2 sm:pl-4">
            <button
              type="button"
              onClick={() => {
                const next = Math.max(1, page - 1);
                setPage(next);
                scrollToPage(next);
              }}
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
              onClick={() => {
                const next = Math.min(numPages ?? page, page + 1);
                setPage(next);
                scrollToPage(next);
              }}
              className="flex size-6 items-center justify-center text-ink-soft hover:text-ink"
              aria-label="Next page"
            >
              <ChevronRight className="size-3.5" />
            </button>
          </div>
          <button
            type="button"
            onClick={() => setChatOpen(true)}
            className="flex size-7 items-center justify-center rounded-sm border border-rule text-ink-soft hover:border-ink hover:text-ink lg:hidden"
            aria-label="Open chat"
          >
            <MessageSquare className="size-3.5" />
          </button>
        </div>
      </header>

      {/* Page strip — horizontally scrollable, above the sheet */}
      {file && (
        <div className="shrink-0 border-b border-rule bg-background/50">
          {showOutline && (
            <div className="border-b border-rule/60">
              <div ref={outlineStripRef} className="flex items-center gap-2 overflow-x-auto px-4 py-2 [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden">
                <span className="label-meta shrink-0 pr-2">Contents</span>
                {outline.length === 0 ? (
                  <span className="font-mono text-[0.68rem] text-ink-faint">
                    {metaQuery.isLoading ? "Loading outline…" : "No outline"}
                  </span>
                ) : (
                  outline.map((o, idx) => (
                    <button
                      key={`${o.title}-${o.page}-${idx}`}
                      type="button"
                      onClick={() => {
                        setPage(o.page);
                        scrollToPage(o.page);
                      }}
                      className={cn(
                        "inline-flex shrink-0 items-center gap-1.5 rounded-full border px-3 py-1 text-xs leading-none transition-colors",
                        page === o.page
                          ? "border-marker bg-marker-soft text-marker"
                          : "border-rule bg-paper text-ink-soft hover:border-ink hover:text-ink",
                      )}
                    >
                      <span className="font-mono text-[0.62rem] text-ink-faint">{o.page}</span>
                      <span className="max-w-[18ch] truncate">{o.title}</span>
                    </button>
                  ))
                )}
              </div>
            </div>
          )}

          <div ref={stripRef} className="flex items-center gap-2 overflow-x-auto px-4 py-3 [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden snap-x snap-mandatory">
            <span className="label-meta shrink-0 pr-1">Pages</span>
            {!thumbnailFileData || numPages == null ? (
              <ThumbnailPlaceholder count={numPages ?? 4} />
            ) : (
              <Document file={thumbnailFileData} options={thumbnailOptions} loading={<ThumbnailPlaceholder count={numPages} />}>
                <div className="flex gap-2">
                  {Array.from({ length: numPages }, (_, i) => i + 1).map((n) => (
                    <button
                      key={n}
                      type="button"
                      onClick={() => {
                        setPage(n);
                        scrollToPage(n);
                      }}
                      className={cn(
                        "group relative flex h-28 shrink-0 snap-start aspect-[3/4] items-center justify-center overflow-hidden rounded-[2px] border bg-paper transition-all",
                        page === n ? "border-marker shadow-sheet" : "border-rule opacity-70 hover:opacity-100",
                      )}
                    >
                      <Page
                        width={STRIP_THUMB_WIDTH}
                        pageNumber={n}
                        renderTextLayer={false}
                        renderAnnotationLayer={false}
                        className="bg-paper [&_canvas]:mx-auto [&_canvas]:block [&_canvas]:max-w-full"
                      />
                      <span className="pointer-events-none absolute right-1 bottom-1 rounded-sm bg-paper/80 px-0.5 font-mono text-[0.55rem] text-ink-faint">
                        {n}
                      </span>
                    </button>
                  ))}
                </div>
              </Document>
            )}
          </div>
        </div>
      )}

      <div className="relative flex min-h-0 min-w-0 flex-1 overflow-hidden">
        {/* Reading sheet */}
        <div ref={scrollRef} onScroll={handleScroll} className="flex min-h-0 min-w-0 flex-1 overflow-x-hidden overflow-y-auto overscroll-contain px-3 sm:px-6 py-8 [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden">
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
              <article className="paper-grain mb-6 bg-paper px-6 sm:px-10 pt-10 pb-8 shadow-sheet">
                <p className="label-meta">Research paper · {file.metadata.content_type}</p>
                <h1 className="mt-3 font-serif text-[1.7rem] leading-[1.15] font-medium text-balance">
                  {displayTitle(file)}
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
                  <span className="font-mono text-[0.62rem] text-ink-faint">Page {String(page).padStart(2, "0")}</span>
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
                  <span className="label-meta">p. {page}</span>
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
                      <ReaderDocument
                        file={file}
                        zoom={zoom}
                        onLoadSuccess={setNumPages}
                        data={fileData}
                        activePage={page}
                        flashedPage={flashedPage}
                      />
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
                  <span className="label-meta">p. {String(page).padStart(2, "0")}</span>
                  <span className="label-meta">{String(page).padStart(2, "0")}</span>
                </div>
              </div>
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
