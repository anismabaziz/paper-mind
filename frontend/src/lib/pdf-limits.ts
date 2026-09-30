import { MAX_MOUNTED_STRIP_THUMBS } from "./page-strip-window";

// Agreed browser limits for PDF reading. Unit tests assert the windows stay
// inside them; the browser spec records the real bundle, canvas, and memory
// numbers against the same budgets.
export const PDF_BROWSER_BUDGETS = {
  // Initial route JavaScript (gzip) without the lazy PDF engine or worker.
  initialJsGzKb: 230,
  // Thumbnails mounted at once in the page strip.
  maxMountedStripThumbs: MAX_MOUNTED_STRIP_THUMBS,
  // Canvases mounted at once in the reading sheet (window behind + ahead).
  maxMountedSheetCanvases: 8,
  // Live copies of one Document's bytes: source + sheet clone + strip clone.
  maxPdfCopiesPerDocument: 3,
  // Long tasks observed while toggling the strip, zooming, or switching.
  maxLongTasks: 0,
} as const;
