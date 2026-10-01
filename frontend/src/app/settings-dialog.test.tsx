import "@testing-library/jest-dom/vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
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

const recoveryError =
  "Settings were not changed. Saved settings were left unchanged. Check the provider details and API key, then try again.";

function renderDialog() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(
    <QueryClientProvider client={queryClient}>
      <SettingsDialog />
    </QueryClientProvider>,
  );
  return queryClient;
}

beforeEach(() => {
  vi.mocked(getSettings).mockResolvedValue(emptySettings);
  vi.mocked(saveSettings).mockResolvedValue(emptySettings);
  vi.mocked(verifySettings).mockResolvedValue({ ok: true, error: null });
  useSettingsUi.setState({ isOpen: true });
});

afterEach(() => {
  cleanup();
  useSettingsUi.setState({ isOpen: false });
  vi.clearAllMocks();
});

describe("SettingsDialog", () => {
  it("shows model capabilities and the cloud data notice", async () => {
    renderDialog();
    await screen.findByRole("dialog");

    fireEvent.change(screen.getByLabelText("Provider"), { target: { value: "google" } });
    fireEvent.change(screen.getByLabelText("Model"), { target: { value: model.id } });

    expect(await screen.findByText("1,048,576 tokens")).toBeInTheDocument();
    expect(screen.getByText("65,536 tokens")).toBeInTheDocument();
    expect(screen.getByText("Structured output")).toBeInTheDocument();
    expect(screen.getByText("Tool use")).toBeInTheDocument();
    expect(screen.getByText(/\$0\.30 input \/ \$2\.50 output per 1M tokens/)).toBeInTheDocument();
    expect(
      screen.getByText(/Questions and retrieved passages leave this machine/),
    ).toBeInTheDocument();
    expect(screen.getByText(/no local-only chat option/)).toBeInTheDocument();
  });

  it("uses the selected model's data location in the notice", async () => {
    const localModel = {
      ...model,
      provider: "local",
      id: "local-chat",
      data_location: "local" as const,
    };
    vi.mocked(getSettings).mockResolvedValue({
      ...emptySettings,
      supported_models: { local: [localModel] },
    });
    renderDialog();
    await screen.findByRole("dialog");

    fireEvent.change(screen.getByLabelText("Provider"), { target: { value: "local" } });
    fireEvent.change(screen.getByLabelText("Model"), { target: { value: localModel.id } });

    expect(screen.getByText(/stay on this machine/)).toBeInTheDocument();
    expect(screen.queryByText(/leave this machine/)).not.toBeInTheDocument();
  });

  it("tests the candidate without saving it", async () => {
    renderDialog();
    await screen.findByRole("dialog");

    fireEvent.change(screen.getByLabelText("Provider"), { target: { value: "google" } });
    fireEvent.change(screen.getByLabelText("Model"), { target: { value: model.id } });
    fireEvent.change(screen.getByLabelText("API key"), {
      target: { value: "sk-candidate" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Test connection" }));

    await waitFor(() => {
      expect(verifySettings).toHaveBeenCalledWith(
        {
          provider: "google",
          model: model.id,
          api_key: "sk-candidate",
        },
        expect.any(AbortSignal),
      );
    });
    expect(saveSettings).not.toHaveBeenCalled();
    expect(screen.getByText("Connection works. Save settings to make it active.")).toBeInTheDocument();
  });

  it("clears state and aborts a candidate when closed mid-request", async () => {
    let requestSignal: AbortSignal | undefined;
    let resolveVerification:
      | ((value: { ok: boolean; error: string | null }) => void)
      | undefined;
    vi.mocked(verifySettings).mockImplementation((_payload, signal) => {
      requestSignal = signal;
      return new Promise((resolve) => {
        resolveVerification = resolve;
      });
    });
    renderDialog();
    await screen.findByRole("dialog");

    fireEvent.change(screen.getByLabelText("Provider"), { target: { value: "google" } });
    fireEvent.change(screen.getByLabelText("Model"), { target: { value: model.id } });
    fireEvent.change(screen.getByLabelText("API key"), {
      target: { value: "sk-in-flight" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Test connection" }));

    await waitFor(() => expect(verifySettings).toHaveBeenCalled());
    expect(screen.getByRole("button", { name: "Close settings" })).not.toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: "Close settings" }));
    expect(requestSignal?.aborted).toBe(true);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

    await act(async () => {
      resolveVerification?.({ ok: true, error: null });
    });
  });

  it("offers a retry when saved settings fail to load", async () => {
    vi.mocked(getSettings).mockRejectedValue(new Error("settings down"));
    renderDialog();

    expect(await screen.findByTestId("settings-retry")).toBeInTheDocument();
    expect(screen.getByText(/Saved settings were left unchanged/)).toBeInTheDocument();

    vi.mocked(getSettings).mockResolvedValue(emptySettings);
    fireEvent.click(screen.getByTestId("settings-retry"));

    await waitFor(() => expect(getSettings).toHaveBeenCalledTimes(2));
  });

  it("keeps the candidate typed after a failed save without restoring anything", async () => {
    vi.mocked(saveSettings).mockRejectedValue(
      Object.assign(new Error("Request failed"), {
        response: { data: { error: "Settings were not changed." } },
      }),
    );
    renderDialog();
    await screen.findByRole("dialog");

    fireEvent.change(screen.getByLabelText("Provider"), { target: { value: "google" } });
    fireEvent.change(screen.getByLabelText("Model"), { target: { value: model.id } });
    fireEvent.change(screen.getByLabelText("API key"), { target: { value: "sk-rejected" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    expect(await screen.findByText("Settings were not changed.")).toBeInTheDocument();
    // The rejected candidate stays editable so it can be corrected, and the
    // masked key of the working configuration is never overwritten with it.
    expect(screen.getByLabelText("API key")).toHaveValue("sk-rejected");
    expect(screen.getByLabelText("Provider")).toHaveValue("google");
    expect(screen.getByLabelText("Model")).toHaveValue(model.id);
  });

  it("keeps the candidate typed after a failed verification", async () => {
    vi.mocked(verifySettings).mockRejectedValue(
      Object.assign(new Error("Request failed"), {
        response: { data: { error: "Could not run the connection test." } },
      }),
    );
    renderDialog();
    await screen.findByRole("dialog");

    fireEvent.change(screen.getByLabelText("Provider"), { target: { value: "google" } });
    fireEvent.change(screen.getByLabelText("Model"), { target: { value: model.id } });
    fireEvent.change(screen.getByLabelText("API key"), { target: { value: "sk-rejected" } });
    fireEvent.click(screen.getByRole("button", { name: "Test connection" }));

    expect(
      await screen.findByText("Could not run the connection test."),
    ).toBeInTheDocument();
    expect(saveSettings).not.toHaveBeenCalled();
    expect(screen.getByLabelText("API key")).toHaveValue("sk-rejected");
  });

  it("clears the API key and feedback when closed", async () => {
    vi.mocked(verifySettings).mockResolvedValue({
      ok: false,
      error: recoveryError,
    });
    renderDialog();
    await screen.findByRole("dialog");

    fireEvent.change(screen.getByLabelText("Provider"), { target: { value: "google" } });
    fireEvent.change(screen.getByLabelText("Model"), { target: { value: model.id } });
    fireEvent.change(screen.getByLabelText("API key"), {
      target: { value: "sk-secret" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Test connection" }));
    expect(
      await screen.findByText(recoveryError),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Close settings" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

    act(() => useSettingsUi.getState().open());
    expect(await screen.findByRole("dialog")).toBeInTheDocument();
    expect(screen.getByLabelText("API key")).toHaveValue("");
    expect(
      screen.queryByText(recoveryError),
    ).not.toBeInTheDocument();
  });
});
