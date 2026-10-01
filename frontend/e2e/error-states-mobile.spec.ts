import { test, expect } from "@playwright/test";
import { beginIsolatedTest } from "./fixtures";

// Small viewport: the library lives in a dialog, and a failed load must
// still name itself and offer a retry there instead of an empty drawer.
test("mobile library names a failed load with a retry", async ({ page }) => {
  await beginIsolatedTest();
  await page.goto("/");

  await page.route("**/files", (route) => route.abort("failed"));
  await page.reload();
  await page.getByRole("button", { name: "Open library" }).click();

  const library = page.getByRole("dialog", { name: "Library" });
  await expect(library).toBeVisible();
  const error = library.getByTestId("library-error");
  await expect(error).toBeVisible();
  await expect(error).toContainText("Could not load your library");
  await expect(library.getByTestId("library-empty")).toHaveCount(0);

  await page.unroute("**/files");
  await library.getByRole("button", { name: /Retry library load/ }).click();
  await expect(
    library.getByTestId("library-list").or(library.getByTestId("library-empty")),
  ).toBeVisible();
  await expect(library.getByTestId("library-error")).toHaveCount(0);
});
