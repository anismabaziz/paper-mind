import { expect, type Page } from "@playwright/test";
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export const BACKEND_URL =
  process.env.BROWSER_BACKEND_URL ?? "http://127.0.0.1:38201";

const here = dirname(fileURLToPath(import.meta.url));
export const SAMPLE_PDF = join(
  here,
  "..",
  "..",
  "backend",
  "evaluation",
  "sample_docs",
  "papermind-rag-primer.pdf",
);

export const WORKING_SETTINGS = {
  provider: "groq",
  model: "openai/gpt-oss-20b",
  api_key: "e2e-working-key",
};

// The deterministic chat adapter always streams this answer when usable
// evidence exists, so specs can assert on it without a model credential.
export const DETERMINISTIC_ANSWER = "grounded in the retrieved PDF text.";

export interface BackendFile {
  name: string;
  original_filename: string;
  is_processed: boolean;
  ingestion: { state: string } | null;
  index: {
    state: string;
    manifest: { index_generation: number } | null;
  };
}

async function backendJson(path: string, init?: RequestInit) {
  const response = await fetch(`${BACKEND_URL}${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(
      `${init?.method ?? "GET"} ${path} -> ${response.status}: ${JSON.stringify(body)}`,
    );
  }
  return body;
}

export const apiGet = (path: string) => backendJson(path);
export const apiPost = (path: string, body?: unknown) =>
  backendJson(path, { method: "POST", body: JSON.stringify(body ?? {}) });
export const apiPut = (path: string, body: unknown) =>
  backendJson(path, { method: "PUT", body: JSON.stringify(body) });
export const apiDelete = (path: string, body: unknown) =>
  backendJson(path, { method: "DELETE", body: JSON.stringify(body) });

export async function seedWorkingSettings() {
  await apiPut("/settings", WORKING_SETTINGS);
}

export async function beginIsolatedTest() {
  // A timed-out spec can die inside its cleanup, so every spec restores the
  // shared server to a known state first instead of trusting the last one.
  await resetRuntime();
  await unfailAll();
  await seedWorkingSettings();
}

export async function clearLibrary() {
  // The library is shared across specs, and the app opens the first ready
  // Document on load, which pulls in the PDF engine. A spec that asserts on
  // what a cold, empty library downloads has to say so itself.
  const data = await apiGet("/files");
  for (const file of data.files as { name: string }[]) {
    await apiDelete(`/files/remove?path=${encodeURIComponent(file.name)}`).catch(
      () => null,
    );
  }
  await waitForLibraryEmpty();
}

export async function waitForLibraryEmpty(timeoutMs = 60_000) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const data = await apiGet("/files");
    if (!(data.files as unknown[]).length) return;
    if (Date.now() > deadline) {
      throw new Error("library never emptied");
    }
    await new Promise((r) => setTimeout(r, 500));
  }
}

export function stagePdf(originalName: string): string {
  const dir = join(here, "..", "test-results", "staged");
  mkdirSync(dir, { recursive: true });
  const staged = join(dir, originalName);
  copyFileSync(SAMPLE_PDF, staged);
  return staged;
}

export async function uploadViaUi(page: Page, originalName: string) {
  const staged = stagePdf(originalName);
  await page.goto("/");
  const rail = page
    .locator(
      '[data-testid="library-list"]:visible, [data-testid="library-empty"]:visible',
    )
    .first();
  // On small screens the rail lives in a closed drawer: open it so the visible
  // upload control can be used. Desktop keeps its inline rail. Wait for the
  // app to mount before deciding, because reading a still-booting page as "no
  // rail" sends a desktop run looking for a drawer button it never renders.
  // isVisible() answers immediately by design, so the wait has to be explicit.
  const inlineRail = await rail
    .waitFor({ state: "visible", timeout: 30_000 })
    .then(() => true)
    .catch(() => false);
  if (!inlineRail) {
    await page.getByRole("button", { name: "Open library" }).click();
  }
  await expect(rail).toBeVisible();
  // The input itself is visually hidden by design; either mounted copy
  // uploads through the same mutation, so the first one is sufficient.
  await page.getByTestId("file-input").first().setInputFiles(staged);
  const file = await findFileByOriginal(originalName);
  return file;
}

export async function findFileByOriginal(
  originalName: string,
  timeoutMs = 30_000,
): Promise<BackendFile> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const data = await apiGet("/files");
    const match = (data.files as BackendFile[]).find(
      (f) => f.original_filename === originalName,
    );
    if (match) return match;
    if (Date.now() > deadline) {
      throw new Error(`uploaded file never appeared: ${originalName}`);
    }
    await new Promise((r) => setTimeout(r, 500));
  }
}

export async function waitForReady(name: string, timeoutMs = 180_000) {
  const deadline = Date.now() + timeoutMs;
  let last = "";
  for (;;) {
    // Force the queue through even if the auto-drain thread is mid-cycle.
    await apiPost("/_test/drain").catch(() => null);
    const data = await apiPost("/file/is-processed", { filename: name });
    last = `${data.is_processed}/${data.ingestion?.state}/${data.index?.state}`;
    if (
      data.is_processed === true &&
      data.ingestion?.state === "ready" &&
      data.index?.state === "ready"
    ) {
      return data;
    }
    if (data.ingestion?.state === "failed") {
      throw new Error(`ingestion failed for ${name}: ${data.ingestion.error_message ?? last}`);
    }
    if (Date.now() > deadline) {
      throw new Error(`index never became ready for ${name}: ${last}`);
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
}

export async function waitForJobState(
  name: string,
  wanted: string[],
  timeoutMs = 120_000,
): Promise<string> {
  const deadline = Date.now() + timeoutMs;
  let state = "";
  for (;;) {
    await apiPost("/_test/drain").catch(() => null);
    const data = await apiGet(`/ingestion-jobs/${encodeURIComponent(name)}`);
    state = data.job?.state ?? "";
    if (wanted.includes(state)) return state;
    if (Date.now() > deadline) {
      throw new Error(`job for ${name} never reached ${wanted}: ${state}`);
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
}

export async function setRuntime(patch: Record<string, Record<string, unknown>>) {
  await apiPost("/_test/runtime", patch);
}

export async function resetRuntime() {
  await apiPost("/_test/runtime/reset");
}

export async function failOp(target: string, operation: string) {
  await apiPost("/_test/fail", { target, operation });
}

export async function unfailOp(target: string, operation: string) {
  await apiPost("/_test/unfail", { target, operation });
}

export async function unfailAll() {
  await apiPost("/_test/unfail-all");
}

export function botMessages(page: Page) {
  // Both the inline pane and the mobile drawer mount a ChatPane; only the
  // visible one carries the conversation under test.
  return page.locator(
    '[data-testid="chat-message"][data-sender="bot"]:visible',
  );
}

export async function selectDocument(page: Page, name: string) {
  const item = page.locator(`[data-testid="library-item-${name}"]:visible`);
  // Selection clicks are ignored while the row shows an active job, and the
  // row polls behind the backend, so wait until indexing settles (ready,
  // stale, failed, or retryable) before clicking.
  await expect(item).toContainText(/Indexed|Stale|Failed|Retry|Cancelled/, {
    timeout: 60_000,
  });
  await item.locator("button").first().click();
}

export async function askQuestion(page: Page, question: string) {
  const input = page.locator('[data-testid="chat-input"]:visible');
  await expect(input).toBeEnabled();
  await input.fill(question);
  await page.locator('[data-testid="chat-send"]:visible').click();
  const answer = botMessages(page).last();
  await expect(answer).toContainText(DETERMINISTIC_ANSWER, { timeout: 60_000 });
  return answer;
}

export async function firstSourceJump(page: Page) {
  const jump = page.locator('[data-testid^="source-jump-"]:visible').first();
  await expect(jump).toBeVisible();
  return jump;
}

export async function citedPageNumber(
  page: Page,
): Promise<{ jump: Awaited<ReturnType<typeof firstSourceJump>>; page: number }> {
  const jump = await firstSourceJump(page);
  // The source row names its page ("[S1] doc · chunk N · p. M").
  const text = await jump.innerText();
  const match = text.match(/p\.\s*(\d+)/);
  return { jump, page: match ? Number(match[1]) : 1 };
}
