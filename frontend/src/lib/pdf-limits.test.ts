import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { PDF_BROWSER_BUDGETS } from "./pdf-limits";

const here = dirname(fileURLToPath(import.meta.url));

function src(relative: string): string {
  return readFileSync(join(here, "..", relative), "utf8");
}

describe("pdf bundle budgets", () => {
  it("declares the browser limits the reader must stay inside", () => {
    expect(PDF_BROWSER_BUDGETS.maxMountedStripThumbs).toBeLessThanOrEqual(24);
    expect(PDF_BROWSER_BUDGETS.maxMountedSheetCanvases).toBeLessThanOrEqual(8);
    expect(PDF_BROWSER_BUDGETS.maxPdfCopiesPerDocument).toBeLessThanOrEqual(3);
    expect(PDF_BROWSER_BUDGETS.initialJsGzKb).toBeLessThanOrEqual(230);
  });

  it("keeps the initial bundle free of the Page renderer", () => {
    // ReaderPane is in the initial route; the pdf.js engine must arrive in a
    // lazy chunk (ReaderDocument / PageStrip) so opening the library never
    // downloads it.
    const readerPane = src("components/papermind/ReaderPane.tsx");
    expect(readerPane).not.toMatch(/from\s+["']react-pdf["']/);
    expect(readerPane).not.toMatch(/from\s+["']pdfjs-dist/);
  });

  it("never renders from a third-party font or character-map host", () => {
    const pane = src("components/papermind/ReaderPane.tsx");
    const document = src("components/papermind/ReaderDocument.tsx");
    for (const content of [pane, document]) {
      expect(content).not.toContain("unpkg.com");
    }
  });
});
