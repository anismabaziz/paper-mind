import { lazy, Suspense, useRef, useState, useEffect, useCallback } from "react";
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
  Loader2,
  RotateCw,
  Square,
} from "lucide-react";
import usePdfStore from "@/store/pdf-state";
import useMobileUi from "@/store/mobile-ui";
import { usePdfFileData } from "@/hooks/usePdfFileData";
import { isDetached } from "@/lib/bytes";
import {
  useFileStatus,
  useFileMeta,
  useDeleteFile,
  useRetryIngestion,
  useCancelIngestion,
} from "@/hooks/useFiles";
import { cn } from "@/lib/utils";
import { displayTitle, ingestionStageLabel, isIngestionActive, isIngestionCancellable, isIngestionRetryable } from "@/types/db";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu";
import { ThumbnailPlaceholder } from "./PageStripPlaceholder";
import { FailureNotice } from "./FailureNotice";

const ReaderDocument = lazy(() => import("./ReaderDocument"));
const PageStrip = lazy(() => import("./PageStrip"));

export function ReaderPane() {
  const { file, citationTarget, setFile } = usePdfStore();
  const scrollRef = useRef<HTMLDivElement>(null);
  const outlineStripRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(100);
  const [showOutline, setShowOutline] = useState(true);
  const [progress, setProgress] = useState(0);
  const [page, setPage] = useState(1);
  const [numPages, setNumPages] = useState<number | null>(null);
  const [flashedPage, setFlashedPage] = useState<number | null>(null);
  const flashTimeoutRef = useRef<number | null>(null);
  // A citation jump that arrived before its page mounted. Retried until the
  // page element exists instead of being dropped mid-render.
  const [pendingCitation, setPendingCitation] = useState<{ page: number; key: number } | null>(null);
  // Single owner of the fetched bytes. Each lazy Renderer (sheet, strip) clones
  // its own copy; unmounting a Renderer releases its clone and switching
  // Documents drops the source, so previous buffers are never retained.
  const { data: fileData, error: fileDataError, reload: reloadFileData } = usePdfFileData(file);

  // Reopening the strip remounts its Document. If the worker already
  // detached the source, refetch fresh bytes so thumbnails reload instead of
  // rendering from a dead view.
  useEffect(() => {
    if (showOutline && fileData && isDetached(fileData)) reloadFileData();
  }, [showOutline, fileData, reloadFileData]);
  const isProgrammaticRef = useRef(false);
  const programmaticTimeoutRef = useRef<number | null>(null);

  const checkProcessedQuery = useFileStatus(file);

  const metaQuery = useFileMeta(file);

  const deleteMutation = useDeleteFile({
    onSuccess: (_data, variables) => {
      if (file?.id === variables.id) setFile(null);
    },
  });
  const deleteError = deleteMutation.error instanceof Error ? deleteMutation.error.message : null;

  const retryIngestion = useRetryIngestion();
  const retryError = retryIngestion.error instanceof Error ? retryIngestion.error.message : null;
  const cancelIngestion = useCancelIngestion();
  const cancelError = cancelIngestion.error instanceof Error ? cancelIngestion.error.message : null;

  const isProcessed = checkProcessedQuery.data?.is_processed ?? false;
  const ingestionJob = checkProcessedQuery.data?.ingestion ?? null;
  const isJobActive = isIngestionActive(ingestionJob?.state);
  const isJobFailed = ingestionJob?.state === "failed";
  const isJobRetryable = isIngestionRetryable(ingestionJob?.state);
  const outline = metaQuery.data?.outline ?? [];
  const metaPageCount = metaQuery.data?.pageCount ?? null;
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
    // reset page when file changes; the strip remounts on file id so its
    // scroll resets without holding the previous Document's scroll node.
    setPage(1);
    setNumPages(null);
    setProgress(0);
    setPendingCitation(null);
    setFlashedPage(null);
    if (outlineStripRef.current) outlineStripRef.current.scrollLeft = 0;
  }, [file?.id]);

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

  // Citation page-jump: ChatPane sets citationTarget → scroll Page N into view.
  // Queued until the page element is mounted so jumps issued while rendering
  // is still in progress land instead of being dropped.
  useEffect(() => {
    if (citationTarget != null) setPendingCitation(citationTarget);
  }, [citationTarget]);

  useEffect(() => {
    if (pendingCitation == null) return;
    const target = pendingCitation.page;
    if (!Number.isFinite(target) || target < 1) {
      setPendingCitation(null);
      return;
    }
    if (numPages != null && target > numPages) {
      setPendingCitation(null);
      return;
    }
    const container = scrollRef.current;
    const mounted = container?.querySelector<HTMLElement>(`[data-page="${target}"]`);
    if (!mounted || !fileData) return; // retry when pages mount
    setPage(target);
    setFlashedPage(target);
    if (flashTimeoutRef.current) window.clearTimeout(flashTimeoutRef.current);
    flashTimeoutRef.current = window.setTimeout(() => setFlashedPage(null), 1700);
    requestAnimationFrame(() => scrollToPage(target));
    setPendingCitation(null);
  }, [pendingCitation, numPages, fileData, scrollToPage]);

  useEffect(() => {
    return () => {
      if (programmaticTimeoutRef.current) window.clearTimeout(programmaticTimeoutRef.current);
      if (flashTimeoutRef.current) window.clearTimeout(flashTimeoutRef.current);
    };
  }, []);

  return (
    <main className="flex min-h-0 min-w-0 flex-1 flex-col bg-canvas" data-testid="reader">
      {retryIngestion.isError && file && (
        <div role="alert" className="border-b border-destructive/40 bg-destructive/5 px-5 py-2">
          <p className="text-xs font-medium text-destructive">Retry failed</p>
          <p className="mt-0.5 text-[0.65rem] text-ink-soft">
            {retryError ?? "The document could not be queued for indexing again."}
          </p>
        </div>
      )}
      {cancelIngestion.isError && file && (
        <div role="alert" className="border-b border-destructive/40 bg-destructive/5 px-5 py-2">
          <p className="text-xs font-medium text-destructive">Cancellation failed</p>
          <p className="mt-0.5 text-[0.65rem] text-ink-soft">
            {cancelError ?? "The indexing job could not be cancelled."}
          </p>
        </div>
      )}
      {file && checkProcessedQuery.isError && checkProcessedQuery.data && (
        <div className="border-b border-rule bg-background px-5 py-2">
          <FailureNotice
            testId="reader-status-stale"
            variant="stale"
            title="Showing the last confirmed state"
            message="This document's indexing status could not be refreshed. Nothing changed on the server."
            actionLabel="Refresh status"
            onAction={() => checkProcessedQuery.refetch()}
          />
        </div>
      )}
      {file && checkProcessedQuery.isError && !checkProcessedQuery.data && (
        <div className="border-b border-destructive/40 bg-destructive/5 px-5 py-2">
          <FailureNotice
            testId="reader-status-error"
            title="Could not check indexing status"
            message={`Questions are paused, not indexing — no confirmed state is available. ${checkProcessedQuery.error instanceof Error ? checkProcessedQuery.error.message : "The status request failed."}`}
            actionLabel="Retry status check"
            onAction={() => checkProcessedQuery.refetch()}
          />
        </div>
      )}
      {deleteMutation.isError && file && (
        <div role="alert" className="border-b border-destructive/40 bg-destructive/5 px-5 py-2">
          <p className="text-xs font-medium text-destructive">Delete failed — the document was kept.</p>
          <p className="mt-0.5 text-[0.65rem] text-ink-soft">{deleteError ?? "Retry deletion."}</p>
          <div className="mt-1.5 flex gap-2">
            <button
              type="button"
              onClick={() => deleteMutation.mutate(file)}
              disabled={deleteMutation.isPending}
              className="border border-ink bg-ink px-2 py-1 font-mono text-[0.6rem] text-paper hover:bg-ink/90 disabled:opacity-40"
            >
              Retry delete
            </button>
            <button
              type="button"
              onClick={() => deleteMutation.reset()}
              className="border border-rule bg-paper px-2 py-1 font-mono text-[0.6rem] hover:border-ink"
            >
              Dismiss
            </button>
          </div>
        </div>
      )}
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
          <div className="min-w-0">
            <p className="truncate font-serif text-[0.95rem] leading-tight">
              {file ? displayTitle(file) : "Document Viewer"}
            </p>
            <p className="label-meta truncate">
              {file ? `${file.metadata.content_type} · ${file.deletion_state === "deleting" ? "Deleting" : file.deletion_state === "delete_failed" ? "Delete failed" : isJobActive ? `${ingestionStageLabel(ingestionJob?.stage ?? "queued")} ${ingestionJob?.progress ?? 0}%` : isJobRetryable ? (isJobFailed ? "Indexing failed" : "Indexing cancelled") : isProcessed ? "Indexed" : "Indexing"}` : "No document selected"}
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
                  aria-label="Document actions"
                  aria-haspopup="menu"
                  title="Document actions"
                >
                  <MoreHorizontal className="size-3.5" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="border-rule bg-paper">
                <DropdownMenuItem
                  className="sm:hidden"
                  onSelect={() => setZoom((z) => Math.max(80, z - 10))}
                >
                  <Minus className="size-3.5" /> Zoom out
                </DropdownMenuItem>
                <DropdownMenuItem
                  className="sm:hidden"
                  onSelect={() => setZoom((z) => Math.min(140, z + 10))}
                >
                  <Plus className="size-3.5" /> Zoom in
                </DropdownMenuItem>
                <DropdownMenuItem
                  className="text-destructive focus:bg-destructive/10 focus:text-destructive"
                  onClick={() => deleteMutation.mutate(file)}
                >
                  <Trash2 className="size-3.5" /> Delete Permanently
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          )}
          {file && (
            <button
              type="button"
              onClick={() => setShowOutline((v) => !v)}
              className={cn(
                "flex size-7 items-center justify-center rounded-sm border transition-colors",
                showOutline ? "border-ink bg-ink text-paper" : "border-rule text-ink-soft hover:border-ink",
              )}
              aria-pressed={showOutline}
              aria-label={showOutline ? "Hide page strip" : "Show page strip"}
              title={showOutline ? "Hide page strip" : "Show page strip"}
            >
              <List className="size-3.5" />
            </button>
          )}

          {/* Zoom lives in the actions menu below the sm breakpoint. With the
              44px touch-target floor, eight controls need more width than a
              phone has, and the overflowing group covered the library and chat
              buttons and swallowed their clicks. */}
          <div className="hidden items-center gap-1 border-l border-rule pl-2 sm:flex sm:pl-4">
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
              data-testid="page-prev"
            >
              <ChevronLeft className="size-3.5" />
            </button>
            <span className="font-mono text-[0.68rem] text-ink-soft" data-testid="page-indicator">
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
              data-testid="page-next"
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
      {file && showOutline && (
        <div className="shrink-0 border-b border-rule bg-background/50">
          <div className="border-b border-rule/60">
            <div ref={outlineStripRef} className="flex items-center gap-2 overflow-x-auto px-4 py-2 [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden">
                <span className="label-meta shrink-0 pr-2">Contents</span>
                {outline.length === 0 ? (
                  metaQuery.isError ? (
                    <span className="flex items-center gap-2">
                      <span role="alert" className="font-mono text-[0.68rem] text-destructive">
                        Outline unavailable —{" "}
                        {metaQuery.error instanceof Error ? metaQuery.error.message : "could not load outline."}
                      </span>
                      <button
                        type="button"
                        onClick={() => metaQuery.refetch()}
                        className="border border-rule bg-paper px-2 py-0.5 font-mono text-[0.62rem] text-ink-soft hover:border-ink hover:text-ink"
                      >
                        Retry
                      </button>
                    </span>
                  ) : (
                    <span className="font-mono text-[0.68rem] text-ink-faint">
                      {metaQuery.isLoading ? "Loading outline…" : "No outline"}
                    </span>
                  )
                ) : (
                  outline.map((o, idx) => (
                    <button
                      key={`${o.title}-${o.page}-${idx}`}
                      type="button"
                      onClick={() => {
                        setPage(o.page);
                        scrollToPage(o.page);
                      }}
                      aria-label={`Jump to ${o.title}, page ${o.page}`}
                      aria-current={page === o.page ? "true" : undefined}
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

          <Suspense fallback={<div className="px-4 py-3"><ThumbnailPlaceholder count={metaPageCount ?? numPages ?? 4} /></div>}>
            <PageStrip
              key={`${file.id}-strip`}
              fileId={file.id}
              source={fileData}
              pageCount={metaPageCount ?? numPages ?? 0}
              activePage={page}
              onSelect={(n) => {
                setPage(n);
                scrollToPage(n);
              }}
              onBufferDetached={reloadFileData}
            />
          </Suspense>
        </div>
      )}

      <div className="relative flex min-h-0 min-w-0 flex-1 overflow-hidden">
        {/* Reading sheet */}
        <div ref={scrollRef} onScroll={handleScroll} className="min-h-0 min-w-0 flex-1 overflow-x-hidden overflow-y-auto overscroll-contain px-3 sm:px-6 py-8 [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden">
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
                    {isJobActive
                      ? `${ingestionStageLabel(ingestionJob?.stage ?? "queued")} · ${ingestionJob?.progress ?? 0}%`
                      : isJobRetryable
                        ? isJobFailed
                          ? "Indexing failed"
                          : "Indexing cancelled"
                        : isProcessed
                          ? "Ready for questions"
                          : "Indexing"}
                  </span>
                  {isJobRetryable && isProcessed && !isJobActive && file && (
                    <button
                      type="button"
                      onClick={() => retryIngestion.mutate(file.name)}
                      disabled={retryIngestion.isPending}
                      className="ml-2 border border-rule bg-paper px-2 py-1 font-mono text-[0.6rem] text-ink-soft hover:border-ink hover:text-ink disabled:opacity-40"
                    >
                      Retry
                    </button>
                  )}
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
                    {fileDataError ? (
                      <div className="grid min-h-[760px] place-items-center bg-white p-6 text-center" data-testid="reader-download-error">
                        <div>
                          <FailureNotice
                            title="Could not download this document"
                            message={`${fileDataError} The document is still in your library — this is a download failure, not a missing document.`}
                          />
                          <div className="mt-3 flex items-center justify-center gap-2">
                            <button
                              type="button"
                              onClick={() => reloadFileData()}
                              className="inline-flex items-center gap-1.5 border border-ink bg-ink px-3 py-1.5 font-mono text-[0.65rem] text-paper hover:bg-ink/90"
                            >
                              <RotateCw className="size-3" /> Retry download
                            </button>
                            <button
                              type="button"
                              onClick={() => setLibraryOpen(true)}
                              className="inline-flex items-center gap-1.5 border border-rule bg-paper px-3 py-1.5 font-mono text-[0.65rem] text-ink-soft hover:border-ink hover:text-ink"
                            >
                              Choose another document
                            </button>
                          </div>
                        </div>
                      </div>
                    ) : (
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
                        pendingPage={pendingCitation?.page ?? null}
                        onRenderError={reloadFileData}
                      />
                    </Suspense>
                    )}
                  </div>
                  {(!isProcessed || isJobActive) && (
                    <div className="absolute inset-3 grid place-items-center bg-paper/70 backdrop-blur-[1px]">
                      <div className="rounded-sm border border-rule bg-paper px-4 py-3 text-center shadow-sheet">
                        <p className="font-mono text-xs font-medium" role="status">
                          {isJobFailed
                            ? "Indexing failed"
                            : isJobRetryable
                              ? "Indexing cancelled"
                              : `${ingestionStageLabel(ingestionJob?.stage ?? "queued")} · ${
                                  ingestionJob?.progress ?? 0
                                }%`}
                        </p>
                        <p className="mt-1 text-xs text-ink-soft">
                          {isJobFailed
                            ? (ingestionJob?.error_message ??
                              "Indexing failed before this document was ready.")
                            : isJobRetryable
                              ? "The indexing job was cancelled. You can retry it when ready."
                              : "Semantic vectors are being generated — chat will unlock when this pass finishes."}
                        </p>
                        {isJobRetryable && file && (
                          <button
                            type="button"
                            onClick={() => retryIngestion.mutate(file.name)}
                            disabled={retryIngestion.isPending}
                            className="mt-3 inline-flex items-center gap-1.5 border border-ink bg-ink px-3 py-1.5 font-mono text-[0.65rem] text-paper hover:bg-ink/90 disabled:opacity-40"
                          >
                            {retryIngestion.isPending ? (
                              <Loader2 className="size-3 animate-spin" />
                            ) : (
                              <RotateCw className="size-3" />
                            )}
                            Retry indexing
                          </button>
                        )}
                        {isJobActive && isIngestionCancellable(ingestionJob?.state) && file && (
                          <button
                            type="button"
                            onClick={() => cancelIngestion.mutate(file.name)}
                            disabled={cancelIngestion.isPending}
                            className="mt-3 ml-2 inline-flex items-center gap-1.5 border border-rule bg-paper px-3 py-1.5 font-mono text-[0.65rem] text-ink-soft hover:border-destructive hover:text-destructive disabled:opacity-40"
                          >
                            {cancelIngestion.isPending ? (
                              <Loader2 className="size-3 animate-spin" />
                            ) : (
                              <Square className="size-3" />
                            )}
                            Cancel indexing
                          </button>
                        )}
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
