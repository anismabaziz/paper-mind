import { useEffect, useMemo, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/TextLayer.css";
import "react-pdf/dist/Page/AnnotationLayer.css";
import { getToken } from "@/services/auth";
import type { File as FileType } from "@/types/db";

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

export default function ReaderDocument({ file, zoom, onLoadSuccess, data: externalData }: Props) {
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
        const token = getToken();
        const headers: Record<string, string> = {};
        if (token) headers.Authorization = `Bearer ${token}`;
        const res = await fetch(file.url, { headers, signal: controller.signal });
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
  // Divide by scale so effective width (pageWidth * scale) fits the outer wrapper
  // without triggering a horizontal scrollbar.
  const sheetWidth = Math.min(880, 7.6 * zoom);
  const canvasPadding = 24; // p-3 *2
  const borderCompensation = 2;
  const scale = zoom / 100;
  const pageWidth = Math.max(320, (sheetWidth - canvasPadding - borderCompensation) / scale);

  function handleLoadSuccess({ numPages: n }: { numPages: number }) {
    setNumPages(n);
    onLoadSuccess(n);
  }

  const pagesToRender = numPages ?? 1;

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
        return (
          <div key={n} id={`page-${n}`} data-page={n} className="scroll-mt-2 bg-white">
            <Page
              pageNumber={n}
              width={pageWidth}
              scale={scale}
              renderTextLayer
              renderAnnotationLayer
              className="bg-white [&_canvas]:mx-auto [&_canvas]:block"
            />
            {n < pagesToRender && <div className="h-3 bg-canvas" />}
          </div>
        );
      })}
    </Document>
  );
}
