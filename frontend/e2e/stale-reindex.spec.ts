import { test, expect } from "@playwright/test";
import {
  DETERMINISTIC_ANSWER,
  apiPost,
  askQuestion,
  beginIsolatedTest,
  botMessages,
  failOp,
  resetRuntime,
  selectDocument,
  setRuntime,
  unfailAll,
  uploadViaUi,
  waitForJobState,
  waitForReady,
} from "./fixtures";

// A stale document blocks chat and offers reindexing. A failed reindex keeps
// the stale state and the prior generation; a successful one makes it ready.
test("stale index blocks chat until a successful reindex", async ({ page }) => {
  await beginIsolatedTest();
  const original = `stale-${Date.now()}.pdf`;
  const file = await uploadViaUi(page, original);
  await waitForReady(file.name);
  await selectDocument(page, file.name);
  await expect(page.getByTestId("chat-input")).toBeEnabled();

  const ready = await apiPost("/file/is-processed", { filename: file.name });
  const firstGeneration = ready.index.manifest.index_generation as number;

  // Simulate a chunking-policy change: the stored index no longer matches.
  await setRuntime({ chunking: { chunk_size_tokens: 999 } });
  try {
    await page.reload();
    await selectDocument(page, file.name);
    await expect(page.getByTestId("stale-index-notice")).toBeVisible();
    await expect(page.getByTestId("chat-input")).toBeDisabled();

    // The HTTP boundary refuses too, before any paid generation could run.
    const refused = await fetch(
      `${process.env.BROWSER_BACKEND_URL ?? "http://127.0.0.1:38201"}/response`,
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          query: "What is this about?",
          filename: file.name,
        }),
      },
    );
    expect(refused.status).toBe(409);
    expect((await refused.json()).category).toBe("index_stale");

    // A failed reindex leaves the document stale on its prior generation.
    await failOp("vector_store", "upsert");
    try {
      await page.getByTestId("chat-reindex").click();
      await waitForJobState(file.name, ["failed"]);
      const stuck = await apiPost("/file/is-processed", {
        filename: file.name,
      });
      expect(stuck.index.state).toBe("stale");
      expect(stuck.index.manifest.index_generation).toBe(firstGeneration);
      await expect(page.getByTestId("stale-index-notice")).toBeVisible();
      await expect(page.getByTestId("chat-input")).toBeDisabled();
    } finally {
      await unfailAll();
    }

    // A successful reindex under the new configuration makes it ready and
    // moves chat to the new generation.
    await page.getByTestId("chat-reindex").click();
    await waitForReady(file.name);
    const replaced = await apiPost("/file/is-processed", {
      filename: file.name,
    });
    expect(replaced.index.state).toBe("ready");
    expect(replaced.index.manifest.index_generation).toBeGreaterThan(
      firstGeneration,
    );
    await expect(page.getByTestId("stale-index-notice")).toBeHidden();
    await askQuestion(page, "What changed after reindexing?");
    await expect(botMessages(page).last()).toContainText(DETERMINISTIC_ANSWER);
  } finally {
    await resetRuntime();
  }
});
