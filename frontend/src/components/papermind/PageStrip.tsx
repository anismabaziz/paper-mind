import { useEffect, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import { cn } from "@/lib/utils";
import { isDetached } from "@/lib/bytes";
import { clonePdfData } from "@/lib/pdf-buffer";
import { pdfDocumentOptions, pdfWorkerSrc } from "@/lib/pdf-assets";
import { ThumbnailPlaceholder } from "./PageStripPlaceholder";
import { FailureNotice } from "./FailureNotice";
import {
  STRIP_THUMB_GAP,
  STRIP_THUMB_WIDTH,
  pageStripWindow,
} from "@/lib/page-strip-window";

pdfjs.GlobalWorkerOptions.workerSrc = pdfWorkerSrc();

// Built once. react-pdf reloads the whole Document whenever `options` changes
// identity, and every load transfers the buffer to the worker — so a fresh
// object each render made the second render ask pdf.js to load bytes that were
// already detached, which rejects and blanks the strip.
const documentOptions = pdfDocumentOptions();

// A detached view renders blank, so it is never worth cloning: the caller
// refetches instead and this reports "no bytes" until fresh ones arrive.
function cloneSource(source: Uint8Array | null) {
  if (!source || isDetached(source)) return null;
  try {
    return clonePdfData(source);
  } catch {
    return null;
  }
}

type Props = {
  fileId: string;
  source: Uint8Array | null;
  pageCount: number;
  activePage: number;
  onSelect: (page: number) => void;
  onBufferDetached: () => void;
};

// Virtualized thumbnail row: only the visible window plus overscan around the
// active Page mounts a canvas. Spacers preserve the scroll width so the strip
// still scrolls like the full Document. Owns exactly one buffer clone; when
// the strip closes or the Document switches, unmounting releases it.
export default function PageStrip({ fileId, source, pageCount, activePage, onSelect, onBufferDetached }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [scrollLeft, setScrollLeft] = useState(0);
  const [viewportWidth, setViewportWidth] = useState(0);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Each Renderer gets its own copy: pdf.js transfers the buffer to the
  // worker and detaches it, so sharing one view would blank the sheet Renderer.
  // Remounting per Document (keyed on fileId) releases the previous clone.
  // The clone remembers its source so re-renders never rebuild it — pdf.js
  // reloads on a new identity and would be handed the spent copy again.
  const [clone, setClone] = useState(() => ({ source, data: cloneSource(source) }));
  useEffect(() => {
    setClone((prev) => (prev.source === source ? prev : { source, data: cloneSource(source) }));
  }, [source]);
  const fileData = clone.data;

  useEffect(() => {
    if (source && isDetached(source)) onBufferDetached();
  }, [source, onBufferDetached]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const update = () => {
      setScrollLeft(el.scrollLeft);
      setViewportWidth(el.clientWidth);
    };
    update();
    el.addEventListener("scroll", update, { passive: true });
    if (typeof ResizeObserver === "undefined") {
      return () => el.removeEventListener("scroll", update);
    }
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => {
      el.removeEventListener("scroll", update);
      observer.disconnect();
    };
  }, []);

  // Keep the active Page thumbnail visible as the sheet scrolls.
  useEffect(() => {
    const container = scrollRef.current;
    if (!container || pageCount === 0) return;
    if (typeof container.scrollTo !== "function") return;
    const target = container.querySelector<HTMLElement>(`[data-strip-page="${activePage}"]`);
    if (!target) return;
    const PADDING = 16;
    const containerRect = container.getBoundingClientRect();
    const targetRect = target.getBoundingClientRect();
    const isFullyVisible = targetRect.left >= containerRect.left + PADDING && targetRect.right <= containerRect.right - PADDING;
    if (!isFullyVisible) {
      const offsetLeft = target.offsetLeft;
      const targetWidth = target.offsetWidth;
      const containerWidth = container.clientWidth;
      const desired = offsetLeft - containerWidth / 2 + targetWidth / 2;
      const maxScroll = container.scrollWidth - containerWidth;
      const clamped = Math.max(0, Math.min(maxScroll, desired));
      container.scrollTo({ left: clamped, behavior: "smooth" });
    }
  }, [activePage, pageCount]);

  const safeCount = Number.isFinite(pageCount) && pageCount > 0 ? Math.floor(pageCount) : 0;
  if (safeCount === 0) return <ThumbnailPlaceholder count={4} />;
  // A failed load is not still loading, and the placeholder is
  // indistinguishable from a working strip. Say what broke and offer the one
  // action that clears it: the Document is remounted on a fresh clone, since
  // the bytes it already handed pdf.js are spent.
  if (loadError) {
    return (
      <div className="px-4 py-3" data-testid="page-strip-error">
        <FailureNotice
          title="Page thumbnails unavailable"
          message={`${loadError} The pages themselves are still readable below.`}
          actionLabel="Retry thumbnails"
          onAction={() => {
            setLoadError(null);
            setClone({ source, data: cloneSource(source) });
          }}
        />
      </div>
    );
  }
  if (!fileData) return <ThumbnailPlaceholder count={Math.min(safeCount, 8)} />;

  const { start, end } = pageStripWindow({
    pageCount: safeCount,
    activePage,
    scrollLeft,
    viewportWidth: viewportWidth || 800,
  });
  const stride = STRIP_THUMB_WIDTH + STRIP_THUMB_GAP;
  const leftSpacer = (start - 1) * stride;
  const rightSpacer = (safeCount - end) * stride;

  return (
    <div ref={scrollRef} className="flex items-center gap-2 overflow-x-auto px-4 py-3 [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden scroll-px-4" data-testid="page-strip">
      <span className="label-meta shrink-0 pr-1">Pages</span>
      <Document
        key={`${fileId}-thumbs`}
        file={fileData}
        options={documentOptions}
        loading={<ThumbnailPlaceholder count={Math.min(safeCount, 8)} />}
        error={null}
        onLoadError={(error: Error) => setLoadError(error.message)}
      >
        <div className="flex gap-2">
          {leftSpacer > 0 && <div aria-hidden style={{ width: leftSpacer }} className="shrink-0" />}
          {Array.from({ length: end - start + 1 }, (_, i) => start + i).map((n) => (
            <button
              key={n}
              type="button"
              data-strip-page={n}
              data-testid="page-strip-thumb"
              onClick={() => onSelect(n)}
              aria-label={`Go to page ${n}`}
              aria-current={activePage === n ? "true" : undefined}
              className={cn(
                "group relative flex h-28 shrink-0 aspect-[3/4] items-center justify-center overflow-hidden rounded-[2px] border bg-paper transition-all",
                activePage === n ? "border-marker shadow-sheet" : "border-rule opacity-70 hover:opacity-100",
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
          {rightSpacer > 0 && <div aria-hidden style={{ width: rightSpacer }} className="shrink-0" />}
        </div>
      </Document>
    </div>
  );
}
