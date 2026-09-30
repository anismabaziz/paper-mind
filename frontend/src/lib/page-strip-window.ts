// The page strip mounts a small window of thumbnails instead of one canvas
// per Page, so a 1,000-page Document costs the same as a short one. The window
// is the union of what the viewport shows and what surrounds the active Page,
// clamped to the Document and capped at the canvas budget.
export const MAX_MOUNTED_STRIP_THUMBS = 24;
export const STRIP_THUMB_WIDTH = 84;
export const STRIP_THUMB_GAP = 8;

export interface PageStripWindowInput {
  pageCount: number;
  activePage: number;
  scrollLeft: number;
  viewportWidth: number;
  thumbWidth?: number;
  gap?: number;
  overscan?: number;
}

export interface PageStripWindow {
  start: number;
  end: number;
}

export function pageStripWindow(input: PageStripWindowInput): PageStripWindow {
  const {
    pageCount,
    viewportWidth,
    scrollLeft,
    thumbWidth = STRIP_THUMB_WIDTH,
    gap = STRIP_THUMB_GAP,
    overscan = 5,
  } = input;
  if (!Number.isFinite(pageCount) || pageCount <= 0) return { start: 1, end: 0 };
  const active = Math.min(Math.max(1, Math.floor(input.activePage) || 1), pageCount);
  if (pageCount <= MAX_MOUNTED_STRIP_THUMBS) return { start: 1, end: pageCount };

  const stride = Math.max(1, thumbWidth + gap);
  const visibleStart = Math.max(1, Math.floor(Math.max(0, scrollLeft) / stride) + 1);
  const visibleCount = Math.max(1, Math.ceil(viewportWidth / stride) + 1);
  const visibleEnd = Math.min(pageCount, visibleStart + visibleCount - 1);

  let start = Math.min(
    Math.max(1, visibleStart - overscan),
    Math.max(1, active - overscan),
  );
  let end = Math.max(
    Math.min(pageCount, visibleEnd + overscan),
    Math.min(pageCount, active + overscan),
  );

  // Cap the window while always keeping the active Page mounted.
  let size = end - start + 1;
  if (size > MAX_MOUNTED_STRIP_THUMBS) {
    const half = Math.floor(MAX_MOUNTED_STRIP_THUMBS / 2);
    start = Math.max(1, Math.min(active - half, pageCount - MAX_MOUNTED_STRIP_THUMBS + 1));
    end = Math.min(pageCount, start + MAX_MOUNTED_STRIP_THUMBS - 1);
    size = end - start + 1;
    void size;
  }
  return { start, end };
}
