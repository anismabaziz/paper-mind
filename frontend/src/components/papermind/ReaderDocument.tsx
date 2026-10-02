import { useEffect, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import { RotateCw } from "lucide-react";
import "react-pdf/dist/Page/TextLayer.css";
import "react-pdf/dist/Page/AnnotationLayer.css";
import type { File as FileType } from "@/types/db";
import { cn } from "@/lib/utils";
import { clonePdfData } from "@/lib/pdf-buffer";
import { isDetached } from "@/lib/bytes";
import { pdfDocumentOptions, pdfWorkerSrc } from "@/lib/pdf-assets";
import { FailureNotice } from "./FailureNotice";

pdfjs.GlobalWorkerOptions.workerSrc = pdfWorkerSrc();

const options = pdfDocumentOptions();

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
  /** Retry the current bytes after a render failure, instead of spinning. */
  onRenderError?: () => void;
};

function PdfLoading({ label = "Loading document…" }: { label?: string }) {
  return (
    <div className="grid h-[760px] place-items-center bg-white">
      <p className="font-mono text-xs text-ink-faint">{label}</p>
    </div>
  );
}

// A Document cannot be rendered from a detached buffer, so a source that pdf.js
// spent is reported as no bytes rather than as a Document that loads blank.
function cloneSource(data: Uint8Array | null): { data: Uint8Array } | null {
  try {
    return clonePdfData(data);
  } catch {
    return null;
  }
}

function PdfError({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="grid h-[760px] place-items-center bg-white p-6 text-center" data-testid="reader-render-error">
      <div>
        <FailureNotice
          title="Could not render this document"
          message={`${message} The document is still in your library — rendering it again usually clears this.`}
        />
        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="mt-3 inline-flex items-center gap-1.5 border border-ink bg-ink px-3 py-1.5 font-mono text-[0.65rem] text-paper hover:bg-ink/90"
          >
            <RotateCw className="size-3" /> Retry rendering
          </button>
        )}
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
  onRenderError,
}: Props) {
  const [internalData, setInternalData] = useState<Uint8Array | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [numPages, setNumPages] = useState<number | null>(null);

  const data = externalData !== undefined ? externalData : internalData;

  // Each Document owns exactly one clone: pdf.js transfers the buffer to its
  // worker and detaches it, so the fetched source is never handed over
  // directly. The clone remembers its source, so a re-render reuses it instead
  // of building a copy of the same bytes — pdf.js reads a new `file` identity
  // as a different Document and reloads what is already on its way.
  const [clone, setClone] = useState<{ source: Uint8Array | null; data: { data: Uint8Array } | null }>(
    () => ({ source: data, data: cloneSource(data) }),
  );
  useEffect(() => {
    setClone((prev) => {
      if (data == null) return prev.source === null ? prev : { source: null, data: null };
      if (prev.source === data && prev.data !== null && !isDetached(prev.data.data)) return prev;
      return { source: data, data: cloneSource(data) };
    });
  }, [data, file.id]);
  const fileData = clone.data;

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

  if (fetchError)
    return (
      <PdfError
        message={fetchError}
        onRetry={() => {
          setFetchError(null);
          onRenderError?.();
        }}
      />
    );
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
