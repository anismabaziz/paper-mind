import { test, expect } from "@playwright/test";
import {
  DETERMINISTIC_ANSWER,
  beginIsolatedTest,
  botMessages,
  failOp,
  selectDocument,
  unfailAll,
  uploadViaUi,
  waitForReady,
} from "./fixtures";

// Failures must never look like empty or loading states: every workflow
// names its failure, keeps the affected context, and offers a recovery.
test("a failed library request is retryable, not an empty library", async ({
  page,
}) => {
  await beginIsolatedTest();
  await page.goto("/");

  // Break the library request at the network edge: the app must say the
  // load failed instead of rendering the empty-library copy.
  await page.route("**/files", (route) => route.abort("failed"));
  await page.reload();
  const error = page.getByTestId("library-error");
  await expect(error).toBeVisible();
  await expect(error).toContainText("Could not load your library");
  await expect(error).toContainText("not an empty library");
  await expect(page.getByTestId("library-empty")).toHaveCount(0);
  await expect(page.getByTestId("library-list")).toHaveCount(0);

  // Recovery runs the same request again: letting traffic through and
  // retrying restores the library instead of stranding the error.
  await page.unroute("**/files");
  await page.getByRole("button", { name: /Retry library load/ }).click();
  await expect(
    page.getByTestId("library-list").or(page.getByTestId("library-empty")),
  ).toBeVisible();
  await expect(page.getByTestId("library-error")).toHaveCount(0);
});

test("an unavailable search reads as unavailable, not a model failure", async ({
  page,
}) => {
  await beginIsolatedTest();
  const original = `error-retrieval-${Date.now()}.pdf`;
  const file = await uploadViaUi(page, original);
  await waitForReady(file.name);
  await selectDocument(page, file.name);

  // The typed refusal the backend returns when the vector store is down
  // (503 "Vector store is unavailable") arrives as a rejected request, not
  // an SSE event: the pane must still name it as unavailable search.
  await page.route("**/response", (route) =>
    route.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({ error: "Vector store is unavailable" }),
    }),
  );
  const input = page.locator('[data-testid="chat-input"]:visible');
  await expect(input).toBeEnabled();
  await input.fill("What is this paper about?");
  await page.locator('[data-testid="chat-send"]:visible').click();

  const answer = botMessages(page).last();
  await expect(answer).toContainText("Search is unavailable");
  await expect(answer).toContainText("no model was asked");
  await expect(page.getByRole("button", { name: "Retry search" })).toBeVisible();

  // The same question works once search recovers: retry sends a new turn.
  await page.unroute("**/response");
  await page.getByRole("button", { name: "Retry search" }).click();
  await expect(botMessages(page).last()).toContainText(DETERMINISTIC_ANSWER, {
    timeout: 60_000,
  });
});

test("a failed document download offers a retry, not an endless loader", async ({
  page,
}) => {
  await beginIsolatedTest();
  const original = `error-download-${Date.now()}.pdf`;
  const file = await uploadViaUi(page, original);
  await waitForReady(file.name);

  // Break only the storage fetch: the Document stays in the library and its
  // index stays ready, so the reader must say the download failed rather
  // than spin on a PDF that will never arrive.
  await page.route("**/storage/**", (route) => route.abort("failed"));
  await selectDocument(page, file.name);

  const failure = page.getByTestId("reader-download-error");
  await expect(failure).toBeVisible();
  await expect(failure).toContainText("Could not download this document");
  await expect(failure).toContainText("not a missing document");
  await expect(page.getByRole("button", { name: /Retry download/ })).toBeVisible();

  // Recovery fetches again: letting traffic through renders the document.
  await page.unroute("**/storage/**");
  await page.getByRole("button", { name: /Retry download/ }).click();
  await expect(page.getByTestId("reader-download-error")).toHaveCount(0);
  await expect(page.getByTestId("page-indicator")).toContainText("/");
});

test("a failed model call keeps the question and offers to ask again", async ({
  page,
}) => {
  await beginIsolatedTest();
  const original = `error-provider-${Date.now()}.pdf`;
  const file = await uploadViaUi(page, original);
  await waitForReady(file.name);
  await selectDocument(page, file.name);

  await failOp("chat", "stream");
  try {
    const input = page.locator('[data-testid="chat-input"]:visible');
    await expect(input).toBeEnabled();
    await input.fill("What is this paper about?");
    await page.locator('[data-testid="chat-send"]:visible').click();

    const answer = botMessages(page).last();
    await expect(answer).toContainText("The model could not answer");
    await expect(page.getByRole("button", { name: "Ask again" })).toBeVisible();
    // A failed answer cites nothing, however much streamed first.
    await expect(answer.locator('[data-testid^="source-jump-"]')).toHaveCount(0);
  } finally {
    await unfailAll();
  }
});
