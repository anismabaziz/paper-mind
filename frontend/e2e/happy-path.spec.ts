import { test, expect } from "@playwright/test";
import {
  DETERMINISTIC_ANSWER,
  apiGet,
  askQuestion,
  beginIsolatedTest,
  botMessages,
  citedPageNumber,
  selectDocument,
  uploadViaUi,
  waitForReady,
} from "./fixtures";

// The critical path: upload a PDF, wait for a ready index, ask a question,
// receive a stream, open the cited page, and find the same answer in history
// after a reload.
test("upload answers with a cited page that survives reload", async ({
  page,
}, testInfo) => {
  await beginIsolatedTest();
  const original = `happy-${Date.now()}.pdf`;
  const question = "What is this paper about?";

  const file = await uploadViaUi(page, original);
  await waitForReady(file.name);
  await expect(page.getByTestId(`library-item-${file.name}`)).toBeVisible();

  await selectDocument(page, file.name);
  await expect(page.getByTestId("chat-input")).toBeEnabled();

  await askQuestion(page, question);

  const { jump, page: cited } = await citedPageNumber(page);
  const padded = String(cited).padStart(2, "0");
  await jump.click();
  await expect(page.getByTestId("page-indicator")).toContainText(
    `${padded} /`,
  );
  const sourceCount = await page
    .locator('[data-testid^="source-jump-"]')
    .count();
  expect(sourceCount).toBeGreaterThan(0);

  await page.screenshot({ path: testInfo.outputPath("happy-path-answer.png") });

  // History is stored turns, not stream residue: reload and reselect.
  await page.reload();
  await expect(
    page
      .getByTestId("library-list")
      .or(page.getByTestId("library-empty")),
  ).toBeVisible();
  await selectDocument(page, file.name);
  await expect(
    page.locator('[data-testid="chat-message"][data-sender="user"]', {
      hasText: question,
    }),
  ).toBeVisible();
  await expect(botMessages(page).last()).toContainText(DETERMINISTIC_ANSWER);
  await expect(page.locator('[data-testid^="source-jump-"]')).toHaveCount(
    sourceCount,
  );

  // The backend agrees the turn was persisted, not just rendered.
  const messages = await apiGet(
    `/messages?filename=${encodeURIComponent(file.name)}`,
  );
  expect(
    messages.messages.some((m: { text: string }) =>
      m.text.includes(DETERMINISTIC_ANSWER),
    ),
  ).toBe(true);
});
