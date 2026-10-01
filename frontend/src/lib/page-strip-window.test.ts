import { describe, expect, it } from "vitest";
import { MAX_MOUNTED_STRIP_THUMBS, pageStripWindow } from "./page-strip-window";

describe("page-strip-window", () => {
  const thumb = { thumbWidth: 84, gap: 8, viewportWidth: 800, overscan: 5 };

  it("mounts every thumbnail for a short Document", () => {
    const window = pageStripWindow({ pageCount: 12, activePage: 1, scrollLeft: 0, ...thumb });
    expect(window).toEqual({ start: 1, end: 12 });
  });

  it("keeps a 50-page strip within the canvas budget around the active Page", () => {
    const window = pageStripWindow({ pageCount: 50, activePage: 25, scrollLeft: 0, ...thumb });
    expect(window.end - window.start + 1).toBeLessThanOrEqual(MAX_MOUNTED_STRIP_THUMBS);
    expect(window.start).toBeLessThanOrEqual(25);
    expect(window.end).toBeGreaterThanOrEqual(25);
  });

  it("keeps a 200-page strip within budget at the start, middle, and end", () => {
    for (const activePage of [1, 100, 200]) {
      const window = pageStripWindow({ pageCount: 200, activePage, scrollLeft: 0, ...thumb });
      expect(window.end - window.start + 1).toBeLessThanOrEqual(MAX_MOUNTED_STRIP_THUMBS);
      expect(window.start).toBeGreaterThanOrEqual(1);
      expect(window.end).toBeLessThanOrEqual(200);
      expect(window.start).toBeLessThanOrEqual(activePage);
      expect(window.end).toBeGreaterThanOrEqual(activePage);
    }
  });

  it("keeps a 1,000-page Document within budget and follows the viewport", () => {
    const startView = pageStripWindow({ pageCount: 1000, activePage: 1, scrollLeft: 0, ...thumb });
    expect(startView.end - startView.start + 1).toBeLessThanOrEqual(MAX_MOUNTED_STRIP_THUMBS);

    // Scrolled far to the right: the window follows the visible range.
    const farScroll = pageStripWindow({
      pageCount: 1000,
      activePage: 900,
      scrollLeft: 80 * 92,
      ...thumb,
    });
    expect(farScroll.end - farScroll.start + 1).toBeLessThanOrEqual(MAX_MOUNTED_STRIP_THUMBS);
    expect(farScroll.start).toBeLessThanOrEqual(900);
    expect(farScroll.end).toBeGreaterThanOrEqual(900);
    expect(farScroll.start).toBeGreaterThan(1);
  });

  it("clamps outside pages instead of mounting placeholders for them", () => {
    const window = pageStripWindow({ pageCount: 50, activePage: 999, scrollLeft: 0, ...thumb });
    expect(window.end).toBe(50);
    expect(window.start).toBeGreaterThanOrEqual(1);
  });
});
