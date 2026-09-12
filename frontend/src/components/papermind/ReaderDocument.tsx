import { useEffect, useMemo, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/TextLayer.css";
import "react-pdf/dist/Page/AnnotationLayer.css";
import type { File as FileType } from "@/types/db";
import { cn } from "@/lib/utils";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).toString();

const options = {
  cMapUrl: `https://unpkg.com/pdfjs-dist@${pdfjs.version}/cmaps/`,
  cMapPacked: true,
  standardFontDataUrl: `https://unpkg.com/pdfjs-dist@${pdfjs.version}/standard_fonts/`,
};

type Props = {
  file: FileType;
  zoom: number;
  onLoadSuccess: (numPages: number) => void;
  data?: Uint8Array | null;
  activePage?: number;
  flashedPage?: number | null;
};

function PdfLoading({ label = "Loading document…" }: { label?: string }) {
  return (
    <div className="grid h-[760px] place-items-center bg-white">
      <p className="font-mono text-xs text-ink-faint">{label}</p>
    </div>
  );
}

function PdfError({ message }: { message: string }) {
  return (
    <div className="grid h-[760px] place-items-center bg-white p-6 text-center">
      <div>
        <p className="font-mono text-xs font-medium text-destructive">Failed to load document</p>
        <p className="mt-1 text-xs text-ink-soft">{message}</p>
      </div>
    </div>
  );
}

export default function ReaderDocument({ file, zoom, onLoadSuccess, data: externalData, activePage, flashedPage }: Props) {
  const [internalData, setInternalData] = useState<Uint8Array | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [numPages, setNumPages] = useState<number | null>(null);

  const data = externalData !== undefined ? externalData : internalData;

  // pdf.js transfers the buffer to the worker, detaching the original.
  // Cloning keeps the state buffer intact so re-renders / StrictMode
  // double-mounts don't hit "ArrayBuffer is detached and could not be cloned".
  const fileData = useMemo(() => (data ? { data: data.slice() } : null), [data]);

  useEffect(() => {
    if (externalData !== undefined) return;
    let cancelled = false;
    const controller = new AbortController();

    async function load() {
      setInternalData(null);
      setFetchError(null);
      try {
        const res = await fetch(file.url, { signal: controller.signal });
        if (!res.ok) throw new Error(`Failed to load PDF (${res.status})`);
        const buf = await res.arrayBuffer();
        if (!cancelled) setInternalData(new Uint8Array(buf));
      } catch (e) {
        if (cancelled || controller.signal.aborted) return;
        setFetchError(e instanceof Error ? e.message : "Failed to load PDF");
      }
    }

    load();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [file.url, externalData]);

  if (fetchError) return <PdfError message={fetchError} />;
  if (!data || !fileData) return <PdfLoading />;

  // Width of the white sheet minus canvas padding (p-3 = 12px each side) and border.
  // Keep in sync with ReaderPane outer width 7.6*zoom cap 880. At 100% => 760px.
  // sheetWidth already encodes zoom, so pageWidth is the inner white width directly.
  const sheetWidth = Math.min(880, 7.6 * zoom);
  const canvasPadding = 24; // p-3 *2
  const borderCompensation = 2;
  const pageWidth = Math.max(320, sheetWidth - canvasPadding - borderCompensation);
  // Visual height preserves scroll position for virtualized placeholders
  const placeholderHeight = Math.round(pageWidth * 1.414);

  function handleLoadSuccess({ numPages: n }: { numPages: number }) {
    setNumPages(n);
    onLoadSuccess(n);
  }

  const pagesToRender = numPages ?? 1;

  // Virtualization: render ±2 pages around active viewport so 100+ page docs
  // do not mount every canvas. Wrappers for all pages remain to preserve
  // scroll height and IntersectionObserver tracking. Flashed citation target
  // is always rendered even if outside the window so the jump lands on a real canvas.
  const activeSafe = activePage != null && Number.isFinite(activePage) ? activePage : 1;
  const windowStart = numPages ? Math.max(1, activeSafe - 2) : 1;
  const windowEnd = numPages ? Math.min(numPages, activeSafe + 2) : 1;

  return (
    <Document
      file={fileData}
      options={options}
      onLoadSuccess={handleLoadSuccess}
      onLoadError={(error: Error) => setFetchError(error.message)}
      loading={<PdfLoading label="Rendering…" />}
      className="bg-white"
    >
      {Array.from({ length: pagesToRender }, (_, i) => {
        const n = i + 1;
        const isInWindow = n >= windowStart && n <= windowEnd;
        const isFlashed = flashedPage === n;
        const shouldRenderPage = isInWindow || isFlashed;
        return (
          <div
            key={n}
            id={`page-${n}`}
            data-page={n}
            className={cn("scroll-mt-2 bg-white flex justify-center", isFlashed && "flash-cite")}
          >
            {shouldRenderPage ? (
              <Page
                pageNumber={n}
                width={pageWidth}
                renderTextLayer
                renderAnnotationLayer
                className={cn("mx-auto bg-white block", isFlashed && "mark-cited")}
              />
            ) : (
              <div
                style={{ height: placeholderHeight }}
                className="grid place-items-center border border-dashed border-rule bg-canvas/30"
                aria-hidden
              >
                <span className="font-mono text-[0.58rem] text-ink-faint">Page {n} · off-screen</span>
              </div>
            )}
            {n < pagesToRender && <div className="h-3 bg-canvas" />}
          </div>
        );
      })}
    </Document>
  );
}
