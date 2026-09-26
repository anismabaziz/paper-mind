import "@testing-library/jest-dom/vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ChatPane } from "./ChatPane";
import { chatStream, reindexFile, StreamProtocolError, type IStreamHandlers } from "@/services/files";
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

/** Renders the pane, asks a question, and runs the given stream script. */
async function ask(run: (handlers: IStreamHandlers) => void) {
  vi.mocked(chatStream).mockImplementation(async (_query, _file, handlers) => {
    run(handlers);
  });
  renderPane();
  const input = await screen.findByPlaceholderText(/Ask this paper something/);
  await act(async () => {
    fireEvent.change(input, { target: { value: "what?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
  });
  return vi.mocked(chatStream).mock.calls[0];
}

function outcomeOf(label: string | RegExp): string | null | undefined {
  return screen.getByText(label).closest("[data-outcome]")?.getAttribute("data-outcome");
}

describe("ChatPane answer outcomes", () => {
  beforeEach(() => {
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
  });

  it("sends the question for the open document", async () => {
    const [query, filename, handlers] = await ask((h) =>
      h.onDone?.({ sources: [], truncated: false, finishReason: "stop" }),
    );

    expect(query).toBe("what?");
    expect(filename).toBe("doc.pdf");
    expect(typeof handlers.onToken).toBe("function");
  });

  it("shows the answer as it streams in", async () => {
    await ask((h) => {
      h.onToken?.("The answer ");
      h.onToken?.("is 42.");
    });

    expect(screen.getByText("The answer is 42.")).toBeInTheDocument();
  });

  it.each([
    ["provider", "The model is unavailable."],
    ["timeout", "This answer took too long."],
    ["empty_output", "The model returned nothing."],
  ] as const)("shows a %s failure as its own outcome", async (category, message) => {
    await ask((h) => h.onProviderError?.(message, category));

    expect(screen.getByText(message)).toBeInTheDocument();
    expect(screen.getByText("Synthesis · failed")).toBeInTheDocument();
    expect(outcomeOf("Synthesis · failed")).toBe(category);
  });

  it("shows a save failure as its own outcome", async () => {
    await ask((h) => h.onPersistenceError?.("The answer could not be saved."));

    expect(outcomeOf("Synthesis · failed")).toBe("persistence");
  });

  it("says why a cancelled answer was not saved, not a loader", async () => {
    await ask((h) => h.onCancelled?.("document deleted"));

    expect(
      screen.getByText(/stopped \(document deleted\) before it finished/),
    ).toBeInTheDocument();
    expect(outcomeOf(/Synthesis · stopped/)).toBe("cancelled");
  });

  it("keeps the answer a cancellation interrupted", async () => {
    await ask((h) => {
      h.onToken?.("Half an answer");
      h.onCancelled?.("document deleted");
    });

    expect(screen.getByText(/Half an answer/)).toBeInTheDocument();
    expect(screen.getByText(/Stopped before this answer was saved/)).toBeInTheDocument();
  });

  it("flags an answer the model cut short", async () => {
    await ask((h) =>
      h.onDone?.({ sources: [], truncated: true, finishReason: "length" }),
    );

    expect(outcomeOf("Synthesis · truncated")).toBe("truncated");
    expect(screen.getByText(/hit its answer limit/)).toBeInTheDocument();
  });

  it("stops the loader when the stream ends with a protocol error", async () => {
    vi.mocked(chatStream).mockRejectedValue(
      new StreamProtocolError("The answer stream ended before it finished."),
    );
    renderPane();
    const input = await screen.findByPlaceholderText(/Ask this paper something/);
    await act(async () => {
      fireEvent.change(input, { target: { value: "what?" } });
      fireEvent.click(screen.getByRole("button", { name: "Send" }));
    });

    expect(
      screen.getByText("The answer stream ended before it finished."),
    ).toBeInTheDocument();
    expect(screen.queryByText(/Reading passages/)).not.toBeInTheDocument();
  });

  it("does not carry a late token into the next document", async () => {
    // A stream that resolves late: its token arrives after the pane moved on.
    let late: ((text: string) => void) | undefined;
    vi.mocked(chatStream).mockImplementation(async (_query, _file, handlers) => {
      late = handlers.onToken;
    });
    renderPane();
    const input = await screen.findByPlaceholderText(/Ask this paper something/);
    await act(async () => {
      fireEvent.change(input, { target: { value: "what?" } });
      fireEvent.click(screen.getByRole("button", { name: "Send" }));
    });

    usePdfStore.getState().setFile({ ...file, id: "file-2", name: "other.pdf" });
    await waitFor(() => expect(screen.getByText("Session Initialized")).toBeInTheDocument());

    act(() => late?.("A token from the old document."));

    expect(screen.queryByText("A token from the old document.")).not.toBeInTheDocument();
  });
});
