import { useEffect, useState } from "react";
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
  page: number;
  zoom: number;
  onLoadSuccess: (numPages: number) => void;
};

export default function ReaderDocument({ file, page, zoom, onLoadSuccess }: Props) {
  const [data, setData] = useState<Uint8Array | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    async function load() {
      setData(null);
      setFetchError(null);
      try {
        const token = getToken();
        const headers: Record<string, string> = {};
        if (token) headers.Authorization = `Bearer ${token}`;
        const res = await fetch(file.url, { headers, signal: controller.signal });
        if (!res.ok) throw new Error(`Failed to load PDF (${res.status})`);
        const buf = await res.arrayBuffer();
        if (!cancelled) setData(new Uint8Array(buf));
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
  }, [file.url]);

  if (fetchError) {
    return (
      <div className="grid h-[760px] place-items-center bg-white p-6 text-center">
        <div>
          <p className="font-mono text-xs font-medium text-destructive">Failed to load document</p>
          <p className="mt-1 text-xs text-ink-soft">{fetchError}</p>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="grid h-[760px] place-items-center bg-white">
        <p className="font-mono text-xs text-ink-faint">Loading document…</p>
      </div>
    );
  }

  // Width of the white sheet minus canvas padding (p-3 = 12px each side) and border.
  // Keep in sync with ReaderPane outer width 7.6*zoom cap 880. At 100% => 760px.
  const sheetWidth = Math.min(880, 7.6 * zoom);
  const canvasPadding = 24; // p-3 *2
  const borderCompensation = 2;
  const pageWidth = Math.max(320, sheetWidth - canvasPadding - borderCompensation);

  return (
    <Document
      file={{ data }}
      options={options}
      onLoadSuccess={({ numPages }: { numPages: number }) => onLoadSuccess(numPages)}
      onLoadError={(error: Error) => setFetchError(error.message)}
      loading={
        <div className="grid h-[760px] place-items-center bg-white">
          <p className="font-mono text-xs text-ink-faint">Rendering…</p>
        </div>
      }
      className="bg-white"
    >
      <Page
        pageNumber={page}
        width={pageWidth}
        renderTextLayer
        renderAnnotationLayer
        className="bg-white [&_canvas]:mx-auto [&_canvas]:block"
      />
    </Document>
  );
}
