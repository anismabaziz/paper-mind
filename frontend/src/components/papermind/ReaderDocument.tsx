import { useEffect, useMemo, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/TextLayer.css";
import "react-pdf/dist/Page/AnnotationLayer.css";
import type { File as FileType } from "@/types/db";
import { cn } from "@/lib/utils";
import { sharedView } from "@/lib/bytes";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).toString();

const options = {
  cMapUrl: `https://unpkg.com/pdfjs-dist@${pdfjs.version}/cmaps/`,
  cMapPacked: true,
  standardFontDataUrl: `https://unpkg.com/pdfjs-dist@${pdfjs.version}/standard_fonts/`,
};

// Pages kept mounted behind / ahead of the active page. Everything else
// renders as a sized placeholder so a 200-page document mounts a handful of
// canvases instead of 200 while scroll offsets stay stable.
const WINDOW_BEHIND = 2;
const WINDOW_AHEAD = 4;

type Props = {
  file: FileType;
  zoom: number;
  onLoadSuccess: (numPages: number) => void;
  data?: Uint8Array | null;
  activePage?: number;
  flashedPage?: number | null;
  // Page a queued citation jump is waiting for. Kept mounted even when it
  // falls outside the active window so the jump can land.
  pendingPage?: number | null;
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

export default function ReaderDocument({
  file,
  zoom,
  onLoadSuccess,
  data: externalData,
  activePage = 1,
  flashedPage = null,
  pendingPage = null,
}: Props) {
  const [internalData, setInternalData] = useState<Uint8Array | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [numPages, setNumPages] = useState<number | null>(null);

  const data = externalData !== undefined ? externalData : internalData;

  // Share one fetch buffer between the sheet and the thumbnail strip via a
  // view over the same ArrayBuffer — no .slice() copy, so a 50 MB file does
  // not become 100–150 MB in JS memory. Note the trade-off: pdf.js transfers
  // the buffer to the worker on load, detaching the shared view afterwards.
  // That is fine for the single-load path (both Documents mount from views
  // created before the transfer); a remount from a detached buffer refetches
  // via the effect below in ReaderPane's loader or this fallback loader.
  const fileData = useMemo(() => (data ? { data: sharedView(data) } : null), [data]);

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
  // Placeholders keep US-Letter aspect so scroll positions survive windowing.
  const placeholderHeight = Math.round(pageWidth * 1.294);

  function handleLoadSuccess({ numPages: n }: { numPages: number }) {
    setNumPages(n);
    onLoadSuccess(n);
  }

  const pagesToRender = numPages ?? 1;
  const windowCenter = pendingPage ?? activePage;
  const windowStart = Math.max(1, windowCenter - WINDOW_BEHIND);
  const windowEnd = Math.min(pagesToRender, windowCenter + WINDOW_AHEAD);

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
        const inWindow = n >= windowStart && n <= windowEnd;
        const flashed = flashedPage === n;
        return (
          <div
            key={n}
            id={`page-${n}`}
            data-page={n}
            className={cn(
              "scroll-mt-2 bg-white flex justify-center transition-shadow",
              flashed && "ring-2 ring-inset ring-marker",
            )}
          >
            {inWindow ? (
              <Page
                pageNumber={n}
                width={pageWidth}
                renderTextLayer
                renderAnnotationLayer={false}
                className={cn("mx-auto bg-white block", flashed && "bg-marker-soft/40")}
              />
            ) : (
              <div
                aria-hidden
                style={{ width: pageWidth, height: placeholderHeight }}
                className={cn("mx-auto bg-white", flashed && "bg-marker-soft/40 ring-2 ring-inset ring-marker")}
              />
            )}
            {n < pagesToRender && <div className="h-3 bg-canvas" />}
          </div>
        );
      })}
    </Document>
  );
}
