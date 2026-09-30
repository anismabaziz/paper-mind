import "@testing-library/jest-dom/vitest";
import axe from "axe-core";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LibraryRail } from "./LibraryRail";
import { ChatPane } from "./ChatPane";
import { chatStream, checkIsProcessed, getFiles, getMessages } from "@/services/files";
import usePdfStore from "@/store/pdf-state";
import type { DocumentIndex, File, IngestionJob } from "@/types/db";

vi.mock("@/services/files", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/services/files")>();
  return {
    ...original,
    chatStream: vi.fn(),
    checkIsProcessed: vi.fn(),
    getFiles: vi.fn(),
    getMessages: vi.fn(),
    reindexFile: vi.fn(),
  };
});

vi.mock("react-pdf", () => ({
  Document: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Page: () => null,
}));

const manifest: NonNullable<DocumentIndex["manifest"]> = {
  content_hash: "abc123",
  parser: "auto",
  parser_version: "pymupdf-docling-page-chunks-v1",
  chunk_size_tokens: 512,
  chunk_overlap_tokens: 50,
  embedding_model: "BAAI/bge-m3",
  embedding_revision: "",
  vector_dimension: 1024,
  sparse_method: "hashed-tf-qdrant-idf-v1",
  sparse_tokenizer_version: "lowercase-regex-stopwords-v1",
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2",
  reranker_revision: "",
  reranker_enabled: false,
  collection_name: "pdf-index",
  collection_schema_version: "named-dense-sparse-v1",
  index_generation: 2,
};

const readyIndex: DocumentIndex = {
  state: "ready",
  manifest,
  runtime_manifest: manifest,
  changes: [],
  change_details: [],
};

const readyJob: IngestionJob = {
  id: "job-1",
  file_id: "file-1",
  filename: "doc.pdf",
  generation: 1,
  state: "ready",
  stage: "ready",
  progress: 100,
  attempt: 1,
  error_category: null,
  error_message: null,
  worker_id: "worker-1",
  created_at: "2026-09-24T10:00:00+00:00",
  updated_at: "2026-09-24T10:00:01+00:00",
  started_at: "2026-09-24T10:00:01+00:00",
  finished_at: null,
  heartbeat_at: "2026-09-24T10:00:01+00:00",
};

const file: File = {
  id: "file-1",
  is_processed: true,
  last_opened_at: null,
  deletion_state: "active",
  deletion_error: null,
  deletion_attempts: 0,
  ingestion: readyJob,
  index: readyIndex,
  metadata: { content_type: "application/pdf", size: 2048 },
  name: "doc.pdf",
  title: "Attention Is All You Need",
  original_filename: "attention.pdf",
  url: "http://localhost/storage/doc.pdf",
};

function renderWithClient(ui: React.ReactNode) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  render(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>);
}

async function axeSerious(container: Element) {
  const results = await axe.run(container, {
    rules: { "color-contrast": { enabled: false } },
  });
  return results.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
}

beforeEach(() => {
  Element.prototype.scrollIntoView = vi.fn();
  vi.mocked(getFiles).mockResolvedValue({ files: [file] });
  vi.mocked(checkIsProcessed).mockResolvedValue({
    is_processed: true,
    ingestion: null,
    index: readyIndex,
  });
  vi.mocked(getMessages).mockResolvedValue({ messages: [] });
  usePdfStore.getState().setFile(file);
});

afterEach(() => {
  cleanup();
  usePdfStore.getState().setFile(null);
  vi.clearAllMocks();
});

describe("LibraryRail keyboard access", () => {
  it("names the search field and reports which tab is current", async () => {
    renderWithClient(<LibraryRail />);
    await screen.findByTestId("library-list");

    expect(screen.getByRole("textbox", { name: "Search library" })).toBeInTheDocument();

    const libraryTab = screen.getByRole("button", { name: /Library/ });
    const recentTab = screen.getByRole("button", { name: /Recent readings/ });
    expect(libraryTab).toHaveAttribute("aria-pressed", "true");
    expect(recentTab).toHaveAttribute("aria-pressed", "false");

    fireEvent.click(recentTab);
    expect(recentTab).toHaveAttribute("aria-pressed", "true");
    expect(libraryTab).toHaveAttribute("aria-pressed", "false");
  });

  it("names the document menu trigger", async () => {
    renderWithClient(<LibraryRail />);
    await screen.findByTestId("library-list");

    const trigger = screen.getByRole("button", {
      name: "Actions for Attention Is All You Need",
    });
    expect(trigger).toHaveAttribute("aria-haspopup", "menu");
  });

  it("has no serious accessibility violations", async () => {
    renderWithClient(<LibraryRail />);
    await screen.findByTestId("library-list");

    expect(await axeSerious(document.body)).toEqual([]);
  });
});

describe("ChatPane keyboard access", () => {
  it("names the question field", async () => {
    renderWithClient(<ChatPane />);
    expect(
      await screen.findByRole("textbox", { name: "Ask this paper a question" }),
    ).toBeInTheDocument();
  });

  it("collapses the cited passages on request", async () => {
    vi.mocked(chatStream).mockImplementation(async (_query, _file, handlers) => {
      handlers.onDone?.({
        sources: [
          {
            content: "Gradient accumulation lets a small batch behave like a large one.",
            document: "doc.pdf",
            chunk_index: 0,
            score: 0.9,
            page: 4,
            source_id: "S1",
            rank: 1,
          },
        ],
        claims: [{ claim: "It is 42.", sources: ["S1"] }],
        grounded: true,
        promptVersion: "grounded-claims-v1",
        truncated: false,
        finishReason: "stop",
      });
    });
    renderWithClient(<ChatPane />);
    const input = await screen.findByRole("textbox", { name: "Ask this paper a question" });
    fireEvent.change(input, { target: { value: "what?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    const toggle = await screen.findByRole("button", { name: "Hide 1 cited passages" });
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    fireEvent.click(toggle);
    expect(
      screen.getByRole("button", { name: "Show 1 cited passages" }),
    ).toHaveAttribute("aria-expanded", "false");
  });

  it("has no serious accessibility violations", async () => {
    renderWithClient(<ChatPane />);
    await screen.findByRole("textbox", { name: "Ask this paper a question" });
    await waitFor(() => expect(screen.queryByText(/Reading passages/)).not.toBeInTheDocument());

    expect(await axeSerious(screen.getByTestId("chat-pane"))).toEqual([]);
  });
});
