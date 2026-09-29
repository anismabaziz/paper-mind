import { test, expect } from "@playwright/test";
import {
  DETERMINISTIC_ANSWER,
  askQuestion,
  beginIsolatedTest,
  botMessages,
  selectDocument,
  uploadViaUi,
  waitForReady,
} from "./fixtures";

// The small-viewport workflow: the library and chat live in dialogs that open
// over the reader, keep focus while open, and return it when closed.
test("mobile drawers carry upload through a cited answer", async ({
  page,
}, testInfo) => {
  await beginIsolatedTest();
  const original = `mobile-${Date.now()}.pdf`;
  const file = await uploadViaUi(page, original);
  await waitForReady(file.name);

  // The rail is a dialog on small screens; the upload helper opened it.
  const library = page.getByRole("dialog", { name: "Library" });
  await expect(library).toBeVisible();
  await expect(
    library.getByTestId(`library-item-${file.name}`),
  ).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath("mobile-library.png") });

  await selectDocument(page, file.name);
  await expect(library).toBeHidden();

  await page.getByRole("button", { name: "Open chat" }).click();
  const chat = page.getByRole("dialog", { name: "Reading companion" });
  await expect(chat).toBeVisible();
  await askQuestion(page, "Summarize key findings");
  await expect(botMessages(page).last()).toContainText(DETERMINISTIC_ANSWER);
  await page.screenshot({ path: testInfo.outputPath("mobile-chat.png") });

  await page.getByRole("button", { name: "Close chat" }).click();
  await expect(chat).toBeHidden();
});
