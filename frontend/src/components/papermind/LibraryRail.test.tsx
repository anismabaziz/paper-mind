import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LibraryRail } from "./LibraryRail";
import { cancelIngestionJob, getFiles, reindexFile, retryIngestionJob } from "@/services/files";
import usePdfStore from "@/store/pdf-state";
import type { DocumentIndex, File, IngestionJob } from "@/types/db";

vi.mock("@/services/files", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/services/files")>();
  return {
    ...original,
    getFiles: vi.fn(),
    retryIngestionJob: vi.fn(),
    cancelIngestionJob: vi.fn(),
    reindexFile: vi.fn(),
  };
});

vi.mock("react-pdf", () => ({
  Document: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Page: () => null,
}));

function makeJob(overrides: Partial<IngestionJob> = {}): IngestionJob {
  return {
    id: "job-1",
    file_id: "file-1",
    filename: "doc.pdf",
    generation: 1,
    state: "running",
    stage: "embedding",
    progress: 45,
    attempt: 1,
    error_category: null,
    error_message: null,
    worker_id: "worker-1",
    created_at: "2026-09-24T10:00:00+00:00",
    updated_at: "2026-09-24T10:00:01+00:00",
    started_at: "2026-09-24T10:00:01+00:00",
    finished_at: null,
    heartbeat_at: "2026-09-24T10:00:01+00:00",
    ...overrides,
  };
}

function makeManifest(overrides: Partial<DocumentIndex["manifest"]> = {}): NonNullable<DocumentIndex["manifest"]> {
  return {
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
    index_generation: 1,
    ...overrides,
  };
}

function makeStaleIndex(overrides: Partial<DocumentIndex> = {}): DocumentIndex {
  const manifest = makeManifest({ chunk_size_tokens: 512 });
  return {
    state: "stale",
    manifest,
    runtime_manifest: makeManifest({ chunk_size_tokens: 1024 }),
    changes: ["chunk_size_tokens"],
    change_details: [
      { field: "chunk_size_tokens", label: "chunk size", indexed: 512, current: 1024 },
    ],
    ...overrides,
  };
}

function makeFile(ingestion: IngestionJob | null, index?: DocumentIndex | null): File {
  return {
    id: "file-1",
    is_processed: false,
    last_opened_at: null,
    deletion_state: "active",
    deletion_error: null,
    deletion_attempts: 0,
    ingestion,
    index: index ?? null,
    metadata: { content_type: "application/pdf", size: 2048 },
    name: "doc.pdf",
    title: "Attention Is All You Need",
    original_filename: "attention.pdf",
    url: "http://localhost/storage/doc.pdf",
  };
}

function renderRail() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  render(
    <QueryClientProvider client={queryClient}>
      <LibraryRail />
    </QueryClientProvider>
  );
  return queryClient;
}

beforeEach(() => {
  vi.mocked(cancelIngestionJob).mockResolvedValue(
    makeJob({ state: "cancelling", stage: "cancelling" })
  );
  vi.mocked(retryIngestionJob).mockResolvedValue(
    makeJob({ state: "queued", stage: "queued", progress: 0, attempt: 2 })
  );
  vi.mocked(reindexFile).mockResolvedValue(
    makeJob({ state: "queued", stage: "queued", progress: 0, generation: 2 })
  );
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("LibraryRail ingestion state", () => {
  it("shows the live stage and progress while a job is running", async () => {
    vi.mocked(getFiles).mockResolvedValue({ files: [makeFile(makeJob())] });
    renderRail();

    expect(await screen.findByText(/Embedding · 45%/)).toBeInTheDocument();
    expect(screen.getByRole("status", { name: "Embedding 45%" })).toBeInTheDocument();
  });

  it("offers cancellation while a job is active", async () => {
    vi.mocked(getFiles).mockResolvedValue({ files: [makeFile(makeJob())] });
    renderRail();

    const cancel = await screen.findByRole("button", {
      name: /Cancel indexing Attention Is All You Need/,
    });
    fireEvent.click(cancel);

    await waitFor(() => {
      expect(cancelIngestionJob).toHaveBeenCalledWith("doc.pdf");
    });
  });

  it("shows the failure message and a retry action", async () => {
    vi.mocked(getFiles).mockResolvedValue({
      files: [
        makeFile(
          makeJob({
            state: "failed",
            stage: "failed",
            progress: 75,
            attempt: 2,
            error_category: "vector_store_unavailable",
            error_message: "The vector store is unavailable.",
          })
        ),
      ],
    });
    renderRail();

    expect(
      await screen.findByText(/The vector store is unavailable\./)
    ).toBeInTheDocument();
    const retry = screen.getByRole("button", {
      name: /Retry indexing Attention Is All You Need/,
    });
    fireEvent.click(retry);

    await waitFor(() => {
      expect(retryIngestionJob).toHaveBeenCalledWith("doc.pdf");
    });
  });

  it("keeps retry available after cancellation", async () => {
    vi.mocked(getFiles).mockResolvedValue({
      files: [
        makeFile(
          makeJob({
            state: "cancelled",
            stage: "cancelled",
            progress: 45,
            error_category: "cancelled",
            error_message: "The indexing job was cancelled.",
          })
        ),
      ],
    });
    renderRail();

    expect(await screen.findByText(/Cancelled/)).toBeInTheDocument();
    const retry = screen.getByRole("button", {
      name: /Retry indexing Attention Is All You Need/,
    });
    fireEvent.click(retry);
    await waitFor(() => expect(retryIngestionJob).toHaveBeenCalledWith("doc.pdf"));
  });

  it("marks a ready job as indexed", async () => {
    vi.mocked(getFiles).mockResolvedValue({
      files: [
        {
          ...makeFile(makeJob({ state: "ready", stage: "ready", progress: 100 })),
          is_processed: true,
        },
      ],
    });
    renderRail();

    expect(await screen.findByText(/Indexed/)).toBeInTheDocument();
  });

  it("opens a failed document so its failure can be inspected", async () => {
    vi.mocked(getFiles).mockResolvedValue({
      files: [
        makeFile(
          makeJob({
            state: "failed",
            stage: "failed",
            progress: 75,
            error_category: "vector_store_unavailable",
            error_message: "The vector store is unavailable.",
          })
        ),
      ],
    });
    renderRail();

    const title = await screen.findByText("Attention Is All You Need");
    fireEvent.click(title.closest("button") as HTMLButtonElement);

    await waitFor(() => {
      expect(usePdfStore.getState().file?.name).toBe("doc.pdf");
    });
  });
});

describe("LibraryRail stale index", () => {
  const readyJob = makeJob({ state: "ready", stage: "ready", progress: 100 });

  it("names the setting that changed and offers a reindex", async () => {
    vi.mocked(getFiles).mockResolvedValue({
      files: [
        { ...makeFile(readyJob, makeStaleIndex()), is_processed: true },
      ],
    });
    renderRail();

    expect(
      await screen.findByText(/Index is stale · reindex to update chunk size/)
    ).toBeInTheDocument();
    expect(screen.getByText("Stale")).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: /Reindex Attention Is All You Need/ })
    );

    await waitFor(() => expect(reindexFile).toHaveBeenCalledWith("doc.pdf"));
  });

  it("keeps a matching index marked as indexed", async () => {
    const manifest = makeManifest();
    vi.mocked(getFiles).mockResolvedValue({
      files: [
        {
          ...makeFile(readyJob, {
            state: "ready",
            manifest,
            runtime_manifest: manifest,
            changes: [],
            change_details: [],
          }),
          is_processed: true,
        },
      ],
    });
    renderRail();

    expect(await screen.findByText(/Indexed/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Reindex/ })).not.toBeInTheDocument();
  });

  it("shows progress instead of a reindex while a job is already running", async () => {
    vi.mocked(getFiles).mockResolvedValue({
      files: [{ ...makeFile(makeJob(), makeStaleIndex()), is_processed: true }],
    });
    renderRail();

    expect(await screen.findByText(/Embedding · 45%/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Reindex/ })).not.toBeInTheDocument();
  });
});
