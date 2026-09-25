import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ChatPane } from "./ChatPane";
import { reindexFile } from "@/services/files";
import { checkIsProcessed, getMessages } from "@/services/files";
import usePdfStore from "@/store/pdf-state";
import type { DocumentIndex, File } from "@/types/db";

vi.mock("@/services/files", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/services/files")>();
  return {
    ...original,
    chatStream: vi.fn(),
    checkIsProcessed: vi.fn(),
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

const file: File = {
  id: "file-1",
  is_processed: true,
  last_opened_at: null,
  deletion_state: "active",
  deletion_error: null,
  deletion_attempts: 0,
  ingestion: null,
  index: null,
  metadata: { content_type: "application/pdf", size: 2048 },
  name: "doc.pdf",
  title: "Attention Is All You Need",
  original_filename: "attention.pdf",
  url: "http://localhost/storage/doc.pdf",
};

function renderPane() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  render(
    <QueryClientProvider client={queryClient}>
      <ChatPane />
    </QueryClientProvider>
  );
  return queryClient;
}

beforeEach(() => {
  // jsdom has no layout, so the scroll-into-view the pane runs on mount is a stub.
  Element.prototype.scrollIntoView = vi.fn();
  usePdfStore.getState().setFile(file);
  vi.mocked(getMessages).mockResolvedValue({ messages: [] });
  vi.mocked(reindexFile).mockResolvedValue({
    id: "job-2",
    file_id: "file-1",
    filename: "doc.pdf",
    generation: 3,
    state: "queued",
    stage: "queued",
    progress: 0,
    attempt: 1,
    error_category: null,
    error_message: null,
    worker_id: null,
    created_at: "2026-09-25T10:00:00+00:00",
    updated_at: null,
    started_at: null,
    finished_at: null,
    heartbeat_at: null,
  });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("ChatPane stale index", () => {
  it("shows the active values and the setting that needs a reindex", async () => {
    vi.mocked(checkIsProcessed).mockResolvedValue({
      is_processed: true,
      ingestion: null,
      index: {
        state: "stale",
        manifest,
        runtime_manifest: { ...manifest, embedding_revision: "9f1c2ab" },
        changes: ["embedding_revision"],
        change_details: [
          { field: "embedding_revision", label: "embedding revision", indexed: "", current: "9f1c2ab" },
        ],
      },
    });
    renderPane();

    expect(await screen.findByText(/Index needs reindexing/)).toBeInTheDocument();
    expect(screen.getByText("embedding revision")).toBeInTheDocument();
    expect(screen.getByText("unpinned → 9f1c2ab")).toBeInTheDocument();
    expect(screen.getByText(/Active index generation 2/)).toBeInTheDocument();
  });

  it("queues a reindex and pauses questions until it succeeds", async () => {
    vi.mocked(checkIsProcessed).mockResolvedValue({
      is_processed: true,
      ingestion: null,
      index: {
        state: "stale",
        manifest,
        runtime_manifest: { ...manifest, chunk_size_tokens: 1024 },
        changes: ["chunk_size_tokens"],
        change_details: [
          { field: "chunk_size_tokens", label: "chunk size", indexed: 512, current: 1024 },
        ],
      },
    });
    renderPane();

    const input = await screen.findByPlaceholderText(/Reindex this paper/);
    expect(input).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: /Reindex this paper/ }));

    await waitFor(() => expect(reindexFile).toHaveBeenCalledWith("doc.pdf"));
  });

  it("leaves questions available when the manifest still matches", async () => {
    vi.mocked(checkIsProcessed).mockResolvedValue({
      is_processed: true,
      ingestion: null,
      index: {
        state: "ready",
        manifest,
        runtime_manifest: manifest,
        changes: [],
        change_details: [],
      },
    });
    renderPane();

    expect(await screen.findByPlaceholderText(/Ask this paper something/)).toBeEnabled();
    expect(screen.queryByText(/Index needs reindexing/)).not.toBeInTheDocument();
  });
});
