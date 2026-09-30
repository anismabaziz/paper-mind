import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import useMobileUi from "@/store/mobile-ui";

vi.mock("@/services/files", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/services/files")>();
  return {
    ...original,
    getFiles: vi.fn(),
    checkIsProcessed: vi.fn(),
    getMessages: vi.fn(),
    getFileMeta: vi.fn(),
  };
});

vi.mock("react-pdf", () => ({
  Document: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Page: () => null,
  pdfjs: { GlobalWorkerOptions: {}, version: "mock" },
}));

import { getFiles } from "@/services/files";

function renderApp() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  render(
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  // jsdom has no layout, so the scroll-into-view the chat pane runs is a stub.
  Element.prototype.scrollIntoView = vi.fn();
  vi.mocked(getFiles).mockResolvedValue({ files: [] });
  useMobileUi.setState({ libraryOpen: false, chatOpen: false });
});

afterEach(() => {
  cleanup();
  useMobileUi.setState({ libraryOpen: false, chatOpen: false });
  vi.clearAllMocks();
});

describe("mobile drawers keyboard access", () => {
  it("traps focus in the library drawer and restores it on Escape", async () => {
    renderApp();
    await screen.findByTestId("library-empty");

    // Browsers focus a toolbar control on click; jsdom does not.
    const opener = screen.getByRole("button", { name: "Open library" });
    opener.focus();
    fireEvent.click(opener);

    const dialog = await screen.findByRole("dialog", { name: "Library" });
    expect(dialog).toHaveFocus();

    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("dialog", { name: "Library" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Open library" })).toHaveFocus();
  });

  it("closes the chat drawer from its backdrop without losing focus", async () => {
    renderApp();
    await screen.findByTestId("library-empty");

    const opener = screen.getByRole("button", { name: "Open chat" });
    opener.focus();
    fireEvent.click(opener);

    const dialog = await screen.findByRole("dialog", { name: "Reading companion" });
    expect(dialog).toHaveFocus();

    fireEvent.click(screen.getByRole("button", { name: "Close chat" }));
    expect(screen.queryByRole("dialog", { name: "Reading companion" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Open chat" })).toHaveFocus();
  });
});
