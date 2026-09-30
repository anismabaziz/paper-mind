import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import PageStrip from "./PageStrip";
import { MAX_MOUNTED_STRIP_THUMBS } from "@/lib/page-strip-window";

vi.mock("react-pdf", () => ({
  Document: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Page: () => null,
  pdfjs: { GlobalWorkerOptions: {}, version: "5.4.296" },
}));

afterEach(() => cleanup());

const source = () => new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);

describe("PageStrip virtualization", () => {
  it("mounts every thumbnail for a short Document", () => {
    render(
      <PageStrip
        fileId="doc-a"
        source={source()}
        pageCount={10}
        activePage={1}
        onSelect={() => {}}
        onBufferDetached={() => {}}
      />,
    );
    expect(screen.getAllByTestId("page-strip-thumb")).toHaveLength(10);
  });

  it("mounts only a window for 50, 200, and 1,000-page Documents", () => {
    for (const pageCount of [50, 200, 1000]) {
      cleanup();
      render(
        <PageStrip
          fileId="doc-big"
          source={source()}
          pageCount={pageCount}
          activePage={Math.floor(pageCount / 2)}
          onSelect={() => {}}
          onBufferDetached={() => {}}
        />,
      );
      const mounted = screen.getAllByTestId("page-strip-thumb").length;
      expect(mounted).toBeLessThanOrEqual(MAX_MOUNTED_STRIP_THUMBS);
      expect(mounted).toBeGreaterThan(0);
    }
  });

  it("always keeps the active Page mounted", () => {
    render(
      <PageStrip
        fileId="doc-big"
        source={source()}
        pageCount={200}
        activePage={150}
        onSelect={() => {}}
        onBufferDetached={() => {}}
      />,
    );
    expect(screen.getByRole("button", { name: "Go to page 150" })).toBeInTheDocument();
  });

  it("renders placeholders when there are no bytes yet", () => {
    cleanup();
    const { container } = render(
      <PageStrip
        fileId="doc-big"
        source={null}
        pageCount={200}
        activePage={1}
        onSelect={() => {}}
        onBufferDetached={() => {}}
      />,
    );
    expect(container.querySelectorAll('[data-testid="page-strip-thumb"]')).toHaveLength(0);
  });
});
