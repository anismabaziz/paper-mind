import { test, expect } from "@playwright/test";
import {
  WORKING_SETTINGS,
  apiGet,
  apiPut,
  beginIsolatedTest,
  failOp,
  unfailAll,
} from "./fixtures";

// Settings verification protects the working key: a failed candidate never
// replaces it, and an abandoned key never lingers in browser state.
test("settings verification keeps the working key on failure", async ({
  page,
}) => {
  await beginIsolatedTest();
  await page.goto("/");
  await expect(
    page.getByTestId("library-list").or(page.getByTestId("library-empty")),
  ).toBeVisible();

  await page.getByRole("button", { name: "Open settings" }).first().click();
  const dialog = page.getByTestId("settings-dialog");
  await expect(dialog).toBeVisible();

  // A working candidate verifies and saves.
  await dialog.getByTestId("settings-provider").selectOption(
    WORKING_SETTINGS.provider,
  );
  await dialog.getByTestId("settings-model").selectOption(WORKING_SETTINGS.model);
  await dialog.getByTestId("settings-api-key").fill("settings-working-key");
  await dialog.getByTestId("settings-test").click();
  await expect(dialog.getByTestId("settings-feedback")).toContainText(
    "Connection works",
  );
  await dialog.getByTestId("settings-save").click();
  await expect(dialog.getByTestId("settings-feedback")).toContainText(
    "Settings saved.",
  );
  const saved = await apiGet("/settings");
  expect(saved.provider).toBe(WORKING_SETTINGS.provider);
  expect(saved.model).toBe(WORKING_SETTINGS.model);

  // A failed candidate is rejected and the saved settings are untouched.
  await failOp("chat", "verify");
  try {
    await dialog.getByTestId("settings-api-key").fill("candidate-bad-key");
    await dialog.getByTestId("settings-save").click();
    await expect(dialog.getByTestId("settings-feedback")).toContainText(
      "not changed",
    );
  } finally {
    await unfailAll();
  }
  const kept = await apiGet("/settings");
  expect(kept.provider).toBe(WORKING_SETTINGS.provider);
  expect(kept.model).toBe(WORKING_SETTINGS.model);
  expect(kept.masked_key).toBe(saved.masked_key);

  // Closing drops the abandoned key from browser state.
  await dialog.getByTestId("settings-api-key").fill("abandoned-key");
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  await page.getByRole("button", { name: "Open settings" }).first().click();
  await expect(page.getByTestId("settings-dialog")).toBeVisible();
  await expect(page.getByTestId("settings-api-key")).toHaveValue("");

  await page.keyboard.press("Escape");
  // Leave the shared server exactly as other specs expect it.
  await apiPut("/settings", WORKING_SETTINGS);
});
