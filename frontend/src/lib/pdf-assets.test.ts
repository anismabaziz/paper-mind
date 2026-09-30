import { describe, expect, it } from "vitest";
import { PDF_CMAP_URL, PDF_STANDARD_FONT_DATA_URL, pdfDocumentOptions } from "./pdf-assets";

describe("pdf-assets", () => {
  it("serves character maps and fonts from the app itself", () => {
    expect(PDF_CMAP_URL).toBe("/cmaps/");
    expect(PDF_STANDARD_FONT_DATA_URL).toBe("/standard_fonts/");
  });

  it("never points rendering assets at a third-party network", () => {
    const options = pdfDocumentOptions();
    expect(options.cMapUrl).toBe("/cmaps/");
    expect(options.standardFontDataUrl).toBe("/standard_fonts/");
    expect(options.cMapPacked).toBe(true);
    for (const url of [options.cMapUrl, options.standardFontDataUrl]) {
      expect(url).not.toContain("http");
      expect(url).not.toContain("unpkg");
    }
  });
});
