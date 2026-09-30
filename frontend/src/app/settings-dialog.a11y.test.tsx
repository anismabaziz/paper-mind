import "@testing-library/jest-dom/vitest";
import axe from "axe-core";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import SettingsDialog from "./settings-dialog";
import {
  getSettings,
  saveSettings,
  verifySettings,
  type IModelCapabilities,
  type ISettings,
} from "@/services/settings";
import useSettingsUi from "@/store/settings-ui";

vi.mock("@/services/settings", () => ({
  getSettings: vi.fn(),
  saveSettings: vi.fn(),
  verifySettings: vi.fn(),
}));

const model: IModelCapabilities = {
  provider: "google",
  id: "gemini-2.5-flash",
  context_window_tokens: 1_048_576,
  max_output_tokens: 65_536,
  structured_output: true,
  tool_use: true,
  input_cost_per_million_usd: 0.3,
  output_cost_per_million_usd: 2.5,
  pricing_tier: "standard",
  data_location: "cloud",
  timeout_seconds: 15,
};

const emptySettings: ISettings = {
  provider: null,
  model: null,
  masked_key: null,
  supported_models: { google: [model] },
};

function renderWithOpener() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(
    <QueryClientProvider client={queryClient}>
      <button type="button" onClick={() => useSettingsUi.getState().open()}>
        open settings
      </button>
      <SettingsDialog />
    </QueryClientProvider>,
  );
}

async function openDialog() {
  // Browsers move focus to a button on click; jsdom does not, so focus it
  // first to mirror what the trap will restore on close.
  const opener = screen.getByRole("button", { name: "open settings" });
  opener.focus();
  fireEvent.click(opener);
  await screen.findByRole("dialog");
}

async function fillValidForm() {
  fireEvent.change(screen.getByLabelText("Provider"), { target: { value: "google" } });
  fireEvent.change(screen.getByLabelText("Model"), { target: { value: model.id } });
  fireEvent.change(screen.getByLabelText("API key"), { target: { value: "sk-key" } });
}

beforeEach(() => {
  vi.mocked(getSettings).mockResolvedValue(emptySettings);
  vi.mocked(saveSettings).mockResolvedValue(emptySettings);
  vi.mocked(verifySettings).mockResolvedValue({ ok: true, error: null });
  useSettingsUi.setState({ isOpen: false });
});

afterEach(() => {
  cleanup();
  useSettingsUi.setState({ isOpen: false });
  vi.clearAllMocks();
});

describe("SettingsDialog keyboard access", () => {
  it("traps Tab inside the dialog", async () => {
    renderWithOpener();
    await openDialog();

    const close = screen.getByRole("button", { name: "Close settings" });
    const test = screen.getByRole("button", { name: "Test connection" });
    expect(close).toHaveFocus();

    test.focus();
    fireEvent.keyDown(document, { key: "Tab", code: "Tab" });
    expect(close).toHaveFocus();

    fireEvent.keyDown(document, { key: "Tab", code: "Tab", shiftKey: true });
    expect(test).toHaveFocus();
  });

  it("closes on Escape and restores focus to the opener", async () => {
    renderWithOpener();
    await openDialog();

    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "open settings" })).toHaveFocus();
  });

  it("does not close on Escape while saving", async () => {
    let resolveSave!: (value: ISettings) => void;
    vi.mocked(saveSettings).mockImplementation(
      () => new Promise((resolve) => { resolveSave = resolve; }),
    );
    renderWithOpener();
    await openDialog();
    await fillValidForm();

    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(saveSettings).toHaveBeenCalled());
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.getByRole("dialog")).toBeInTheDocument();

    resolveSave(emptySettings);
    await screen.findByText("Settings saved.");
  });

  it("closes on backdrop click and restores focus", async () => {
    renderWithOpener();
    await openDialog();

    fireEvent.mouseDown(screen.getByTestId("settings-backdrop"));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "open settings" })).toHaveFocus();
  });

  it("submits from the keyboard without a pointer", async () => {
    renderWithOpener();
    await openDialog();
    await fillValidForm();

    // Enter in the form submits it, the same as activating Save.
    fireEvent.submit(screen.getByTestId("settings-dialog").querySelector("form")!);
    await waitFor(() => expect(saveSettings).toHaveBeenCalled());
    expect(await screen.findByText("Settings saved.")).toBeInTheDocument();
  });

  it("keeps loading understandable without animation", async () => {
    let resolveSettings!: (value: ISettings) => void;
    vi.mocked(getSettings).mockImplementation(
      () => new Promise((resolve) => { resolveSettings = resolve; }),
    );
    renderWithOpener();
    fireEvent.click(screen.getByRole("button", { name: "open settings" }));
    await screen.findByRole("dialog");

    // The spinner is decorative; the text carries the state when motion is off.
    const loading = screen.getByRole("status", { name: "Loading settings" });
    expect(loading).toHaveTextContent("Loading settings…");

    resolveSettings(emptySettings);
    await screen.findByLabelText("Provider");
  });

  it("announces feedback to assistive technology", async () => {
    renderWithOpener();
    await openDialog();
    await fillValidForm();

    fireEvent.click(screen.getByRole("button", { name: "Test connection" }));
    const feedback = await screen.findByTestId("settings-feedback");
    expect(feedback).toHaveAttribute("role", "status");
  });
});

describe("SettingsDialog automated checks", () => {
  it("has no serious accessibility violations", async () => {
    renderWithOpener();
    await openDialog();
    await fillValidForm();

    const results = await axe.run(screen.getByTestId("settings-dialog"), {
      rules: { "color-contrast": { enabled: false } },
    });
    const serious = results.violations.filter(
      (v) => v.impact === "serious" || v.impact === "critical",
    );
    expect(serious).toEqual([]);
  });
});
