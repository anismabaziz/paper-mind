import { useCallback, useEffect, useRef, useState } from "react";

type Args = {
  /** The open Document, so navigation resets when a different one opens. */
  fileId: string | null;
  zoom: number;
  /** True once the bytes and the pages they render exist. */
  pagesReady: boolean;
  citationTarget: { page: number; key: number } | null;
};

/**
 * Where the reader is: which page is showing, how far down the sheet is, and
 * where a citation asked it to go.
 *
 * A jump is queued rather than dropped. The page it names may not be mounted
 * yet — pdf.js is still loading, or the sheet is on a page that has not been
 * rendered — and a jump that arrives too early is exactly the one a reader
 * pressing a citation needs to land.
 */
export function useReaderNavigation({ fileId, zoom, pagesReady, citationTarget }: Args) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const outlineStripRef = useRef<HTMLDivElement>(null);
  const [page, setPage] = useState(1);
  const [numPages, setNumPages] = useState<number | null>(null);
  const [progress, setProgress] = useState(0);
  const [flashedPage, setFlashedPage] = useState<number | null>(null);
  const [pendingCitation, setPendingCitation] = useState<{ page: number; key: number } | null>(null);

  const flashTimeoutRef = useRef<number | null>(null);
  const isProgrammaticRef = useRef(false);
  const programmaticTimeoutRef = useRef<number | null>(null);

  const scrollToPage = useCallback((pageNum: number) => {
    const container = scrollRef.current;
    if (!container) return;
    const target = container.querySelector<HTMLElement>(`[data-page="${pageNum}"]`);
    if (!target) return;
    isProgrammaticRef.current = true;
    if (programmaticTimeoutRef.current) window.clearTimeout(programmaticTimeoutRef.current);
    // Compute the offset relative to the scroll container, which includes the
    // height of the title sheet above the first page.
    const containerRect = container.getBoundingClientRect();
    const targetRect = target.getBoundingClientRect();
    const top = targetRect.top - containerRect.top + container.scrollTop;
    container.scrollTo({ top, behavior: "smooth" });
    programmaticTimeoutRef.current = window.setTimeout(() => {
      isProgrammaticRef.current = false;
    }, 800);
  }, []);

  useEffect(() => {
    // The strip is remounted per Document, so its scroll resets on its own;
    // this only has to drop the previous Document's position.
    setPage(1);
    setNumPages(null);
    setProgress(0);
    setPendingCitation(null);
    setFlashedPage(null);
    if (outlineStripRef.current) outlineStripRef.current.scrollLeft = 0;
  }, [fileId]);

  useEffect(() => {
    if (numPages && page > numPages) setPage(numPages);
  }, [numPages, page]);

  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const ratio = el.scrollTop / Math.max(1, el.scrollHeight - el.clientHeight);
    setProgress(Math.round(ratio * 100));
  }, []);

  // The active page is whichever one occupies the most of the viewport, which
  // is what a reader scrolling would say they are looking at. A scroll the app
  // started itself is ignored: it would otherwise land on the page it is
  // scrolling towards and report the reader as already there.
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
            if (!best || entry.intersectionRatio > best.ratio) {
              best = { page: pg, ratio: entry.intersectionRatio };
            }
          }
          if (best) setPage(best.page);
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
  }, [numPages, fileId, zoom, pagesReady]);

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
    const mounted = scrollRef.current?.querySelector<HTMLElement>(`[data-page="${target}"]`);
    // No page element yet means the pages are still mounting: try again when
    // they do rather than dropping the jump.
    if (!mounted || !pagesReady) return;
    setPage(target);
    setFlashedPage(target);
    if (flashTimeoutRef.current) window.clearTimeout(flashTimeoutRef.current);
    flashTimeoutRef.current = window.setTimeout(() => setFlashedPage(null), 1700);
    requestAnimationFrame(() => scrollToPage(target));
    setPendingCitation(null);
  }, [pendingCitation, numPages, pagesReady, scrollToPage]);

  useEffect(() => {
    return () => {
      if (programmaticTimeoutRef.current) window.clearTimeout(programmaticTimeoutRef.current);
      if (flashTimeoutRef.current) window.clearTimeout(flashTimeoutRef.current);
    };
  }, []);

  return {
    page,
    setPage,
    numPages,
    setNumPages,
    progress,
    flashedPage,
    pendingPage: pendingCitation?.page ?? null,
    scrollToPage,
    handleScroll,
    scrollRef,
    outlineStripRef,
  };
}