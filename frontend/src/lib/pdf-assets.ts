// PDF runtime assets are served from the app itself so reading never depends
// on a third-party network. The character maps and standard fonts are vendored
// under public/cmaps and public/standard_fonts (see frontend README / ADR).
export const PDF_CMAP_URL = "/cmaps/";
export const PDF_STANDARD_FONT_DATA_URL = "/standard_fonts/";

export interface PdfDocumentOptions {
  cMapUrl: string;
  cMapPacked: boolean;
  standardFontDataUrl: string;
}

export function pdfDocumentOptions(): PdfDocumentOptions {
  return {
    cMapUrl: PDF_CMAP_URL,
    cMapPacked: true,
    standardFontDataUrl: PDF_STANDARD_FONT_DATA_URL,
  };
}

export function pdfWorkerSrc(): string {
  return new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).toString();
}
