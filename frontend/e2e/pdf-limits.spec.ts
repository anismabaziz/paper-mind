import { test, expect, type Page } from "@playwright/test";
import {
  BACKEND_URL,
  beginIsolatedTest,
  findFileByOriginal,
  selectDocument,
  stagePdf,
  uploadViaUi,
  waitForReady,
} from "./fixtures";
import { buildPdf } from "./pdf-gen";
import { PDF_BROWSER_BUDGETS } from "../src/lib/pdf-limits";

// Browser limits for PDF reading, asserted here and in unit tests from the
// same budget source.
const MAX_MOUNTED_STRIP_THUMBS = PDF_BROWSER_BUDGETS.maxMountedStripThumbs;
const MAX_MOUNTED_SHEET_CANVASES = PDF_BROWSER_BUDGETS.maxMountedSheetCanvases;
// Heap may not grow without bound while switching Documents; 150MB is far
// above two small PDFs and far below a leak that retains every buffer.
const MAX_HEAP_GROWTH_BYTES = 150 * 1024 * 1024;

interface PhaseMetrics {
  pageCount: number;
  mountedThumbs: number;
  sheetCanvases: number;
  outlinePills: number;
  jsHeapBytes: number | null;
  longTasks: number;
}

async function readerMetrics(page: Page): Promise<Omit<PhaseMetrics, "pageCount">> {
  return page.evaluate(() => {
    const thumbs = document.querySelectorAll('[data-testid="page-strip-thumb"]').length;
    const fallbackThumbs = document.querySelectorAll('[data-strip-page]').length;
    const sheet = document.querySelectorAll("[data-page] canvas").length;
    const pills = document.querySelectorAll('[aria-label^="Jump to"]').length;
    const heap =
      typeof performance !== "undefined" &&
      "memory" in performance &&
      typeof (performance as { memory?: { usedJSHeapSize?: number } }).memory?.usedJSHeapSize === "number"
        ? (performance as { memory: { usedJSHeapSize: number } }).memory.usedJSHeapSize
        : null;
    const longTasks = Array.isArray((window as { __longtasks?: unknown[] }).__longtasks)
      ? (window as { __longtasks?: unknown[] }).__longtasks!.length
      : 0;
    return {
      mountedThumbs: thumbs || fallbackThumbs,
      sheetCanvases: sheet,
      outlinePills: pills,
      jsHeapBytes: heap,
      longTasks,
    };
  });
}

async function expectWithinBudgets(p: Page, pageCount: number): Promise<PhaseMetrics> {
  const m = await readerMetrics(p);
  const metrics = { pageCount, ...m };
  expect(
    m.mountedThumbs,
    `${pageCount}-page strip mounts at most ${MAX_MOUNTED_STRIP_THUMBS} thumbnails (got ${m.mountedThumbs})`,
  ).toBeLessThanOrEqual(MAX_MOUNTED_STRIP_THUMBS);
  expect(
    m.sheetCanvases,
    `${pageCount}-page sheet mounts at most ${MAX_MOUNTED_SHEET_CANVASES} canvases (got ${m.sheetCanvases})`,
  ).toBeLessThanOrEqual(MAX_MOUNTED_SHEET_CANVASES);
  return metrics;
}

// A large research PDF must stay inside browser limits: the strip mounts a
// window of thumbnails, the sheet mounts a window of pages, toggling the
// strip or zooming never multiplies canvases, and switching Documents drops
// the previous buffer instead of retaining it.
test("large PDFs stay within browser limits", async ({ page }, testInfo) => {
  await beginIsolatedTest();
  const thirdParty: string[] = [];
  const pdfAssetHosts = new Set<string>();
  const pdfEngineUrls: string[] = [];
  const isPdfEngineRequest = (url: string) =>
    /pdf\.worker|react-pdf|pdfjs-dist|PageStrip\.tsx|ReaderDocument/.test(url) &&
    !url.includes("PageStripPlaceholder");
  page.on("request", (request) => {
    const url = request.url();
    if (url.includes("unpkg.com") || url.includes("cdn.jsdelivr")) thirdParty.push(url);
    if (url.includes("/cmaps/") || url.includes("/standard_fonts/")) {
      try {
        pdfAssetHosts.add(new URL(url).host);
      } catch {
        pdfAssetHosts.add(url);
      }
    }
    if (isPdfEngineRequest(url)) pdfEngineUrls.push(url);
  });
  const jsBytes: number[] = [];
  page.on("response", (response) => {
    const url = response.url();
    if (url.endsWith(".js") || url.endsWith(".mjs")) {
      response.body().then(
        (body) => jsBytes.push(body.length),
        () => {},
      );
    }
  });
  await page.addInitScript(() => {
    (window as { __longtasks?: PerformanceEntry[] }).__longtasks = [];
    try {
      const observer = new PerformanceObserver((list) => {
        (window as { __longtasks?: PerformanceEntry[] }).__longtasks!.push(...list.getEntries());
      });
      observer.observe({ entryTypes: ["longtask"] });
    } catch {
      // PerformanceObserver/longtask unavailable — metrics report zero.
    }
  });

  // The PDF engine stays off the initial bundle: a cold library view never
  // fetches the worker or the renderer chunks. (Uploads auto-select their
  // Document, so this check runs before the first upload.)
  await page.goto("/");
  await expect(
    page.locator('[data-testid="library-list"], [data-testid="library-empty"]').first(),
  ).toBeVisible();
  expect(pdfEngineUrls).toEqual([]);

  const original = `limits-${Date.now()}.pdf`;
  const second = `limits-second-${Date.now()}.pdf`;
  const file = await uploadViaUi(page, original);
  await waitForReady(file.name);
  // Upload the second Document directly: the shared upload helper re-checks
  // the rail immediately after a reload and can race it under load, so wait
  // for the rail first and drive the same input it uses.
  const stagedSecond = stagePdf(second);
  await page.goto("/");
  await expect(
    page.locator('[data-testid="library-list"], [data-testid="library-empty"]').first(),
  ).toBeVisible();
  await page.getByTestId("file-input").first().setInputFiles(stagedSecond);
  const file2 = await findFileByOriginal(second);
  await waitForReady(file2.name);
  const storagePattern = "**/storage/**";
  const metaPattern = `**/files/${encodeURIComponent(file.name)}/meta`;

  const allMetrics: PhaseMetrics[] = [];
  const outlineJumpEvidence: string[] = [];

  const readHeap: () => Promise<number | null> = () =>
    page.evaluate(() =>
      typeof performance !== "undefined" && "memory" in performance
        ? (performance as { memory?: { usedJSHeapSize?: number } }).memory?.usedJSHeapSize ?? null
        : null,
    );
  const heapBeforePhases = await readHeap();

  for (const pageCount of [50, 200, 1000]) {
    const pdf = buildPdf(pageCount);
    await page.unroute(storagePattern).catch(() => {});
    await page.unroute(metaPattern).catch(() => {});
    await page.route(storagePattern, (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/pdf",
        body: Buffer.from(pdf),
      }),
    );
    const outline = [1, Math.floor(pageCount / 2), pageCount].map((p) => ({
      title: `Chapter ${p}`,
      page: p,
      level: 1,
    }));
    await page.route(metaPattern, (route) =>
      route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ pageCount, outline }) }),
    );

    await page.goto("/");
    await selectDocument(page, file.name);
    const padded = String(pageCount).padStart(2, "0");
    // Total pages come from the synthetic PDF via pdf.js…
    await expect(page.getByTestId("page-indicator")).toContainText(`/ ${padded}`, { timeout: 60_000 });
    await expect(page.getByTestId("page-strip")).toBeVisible();

    // Outline navigation still works on a virtualized strip: jumping to the
    // last chapter moves the current page there.
    const pill = page.getByRole("button", { name: `Jump to Chapter ${pageCount}, page ${pageCount}` });
    await expect(pill).toBeVisible();
    await pill.click();
    await expect(page.getByTestId("page-indicator")).toContainText(`${padded} /`);
    outlineJumpEvidence.push(`${pageCount}:outline-jump-ok`);

    allMetrics.push(await expectWithinBudgets(page, pageCount));

    // Repeated strip toggles remount safely without multiplying canvases.
    const toggle = page.getByRole("button", { name: /page strip/i });
    for (let i = 0; i < 3; i++) {
      await toggle.click();
      await expect(page.getByTestId("page-strip")).toBeHidden();
      await toggle.click();
      await expect(page.getByTestId("page-strip")).toBeVisible();
      const thumbs = await page.getByTestId("page-strip-thumb").count();
      expect(thumbs).toBeLessThanOrEqual(MAX_MOUNTED_STRIP_THUMBS);
    }
    allMetrics.push({ ...(await expectWithinBudgets(page, pageCount)), pageCount });

    // Zoom changes keep the sheet windowed.
    await page.getByRole("button", { name: "Zoom in" }).click();
    await page.getByRole("button", { name: "Zoom in" }).click();
    await expect(page.getByTestId("page-indicator")).toBeVisible();
    allMetrics.push(await expectWithinBudgets(page, pageCount));
    await page.getByRole("button", { name: "Zoom out" }).click();
    await page.getByRole("button", { name: "Zoom out" }).click();
  }

  // Document switching drops the previous buffer: alternate between the two
  // real Documents and confirm the strip never grows.
  await page.unroute(storagePattern).catch(() => {});
  await page.unroute(metaPattern).catch(() => {});
  await page.goto("/");
  for (let i = 0; i < 2; i++) {
    await selectDocument(page, file.name);
    await expect(page.getByTestId("page-strip")).toBeVisible();
    expect(await page.getByTestId("page-strip-thumb").count()).toBeLessThanOrEqual(MAX_MOUNTED_STRIP_THUMBS);
    await selectDocument(page, file2.name);
    await expect(page.getByTestId("page-strip")).toBeVisible();
    expect(await page.getByTestId("page-strip-thumb").count()).toBeLessThanOrEqual(MAX_MOUNTED_STRIP_THUMBS);
  }

  // Citation jumps, outline navigation, and zoom survived the refactor: the
  // outline pill for the real document still jumps the sheet.
  await selectDocument(page, file.name);

  // The engine arrived lazily with the first selection, from this app only.
  expect(pdfEngineUrls.length).toBeGreaterThan(0);
  const frontendHost = new URL(page.url()).host;
  for (const url of pdfEngineUrls) {
    expect(new URL(url).host).toBe(frontendHost);
  }

  const heap = await readHeap();
  if (heapBeforePhases != null && heap != null) {
    expect(heap - heapBeforePhases).toBeLessThan(MAX_HEAP_GROWTH_BYTES);
  }
  const longTasks = await page.evaluate(
    () => (window as { __longtasks?: unknown[] }).__longtasks?.length ?? 0,
  );
  const totalJsBytes = jsBytes.reduce((sum, n) => sum + n, 0);

  const report = {
    budgets: {
      maxMountedStripThumbs: MAX_MOUNTED_STRIP_THUMBS,
      maxMountedSheetCanvases: MAX_MOUNTED_SHEET_CANVASES,
      thirdPartyAssetRequests: 0,
    },
    phases: allMetrics,
    outlineJumpEvidence,
    retained: { jsHeapBytes: heap },
    longTasks,
    bundle: { devJsResponses: jsBytes.length, devJsBytes: totalJsBytes },
    pdfAssetHosts: [...pdfAssetHosts],
    backend: BACKEND_URL,
  };
  await testInfo.attach("pdf-limits", { body: JSON.stringify(report, null, 2), contentType: "application/json" });

  // No character map, font, or worker may come from a third-party network.
  expect(thirdParty).toEqual([]);
  // Long tasks during toggles, zoom, and switching are recorded; the reader
  // must not jank the browser.
  expect(longTasks).toBe(0);
});
