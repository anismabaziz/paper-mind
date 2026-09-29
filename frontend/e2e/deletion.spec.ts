import { test, expect } from "@playwright/test";
import { apiGet, beginIsolatedTest, uploadViaUi, waitForReady } from "./fixtures";

// Deleting a document removes it and returns the library to a state the
// backend confirms, not just a row hidden in the browser.
test("deletion removes the document and the library reflects it", async ({
  page,
}) => {
  await beginIsolatedTest();
  const original = `delete-${Date.now()}.pdf`;
  const file = await uploadViaUi(page, original);
  await waitForReady(file.name);
  await expect(page.getByTestId(`library-item-${file.name}`)).toBeVisible();

  const before = (await apiGet("/files")).files as { name: string }[];
  expect(before.some((f) => f.name === file.name)).toBe(true);

  await page.getByTestId(`document-menu-${file.name}`).click();
  await page.getByTestId(`delete-paper-${file.name}`).click();
  await expect(page.getByTestId(`library-item-${file.name}`)).toBeHidden({
    timeout: 30_000,
  });

  const after = (await apiGet("/files")).files as { name: string }[];
  expect(after.some((f) => f.name === file.name)).toBe(false);
  expect(after.length).toBe(before.length - 1);

  // The removal survives a reload: the library shows its updated state.
  await page.reload();
  await expect(
    page.getByTestId("library-list").or(page.getByTestId("library-empty")),
  ).toBeVisible();
  await expect(page.getByTestId(`library-item-${file.name}`)).toHaveCount(0);
});
