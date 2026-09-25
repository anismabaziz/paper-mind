import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LibraryRail } from "./LibraryRail";
import { cancelIngestionJob, getFiles, retryIngestionJob } from "@/services/files";
import usePdfStore from "@/store/pdf-state";
import type { File, IngestionJob } from "@/types/db";

vi.mock("@/services/files", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/services/files")>();
  return {
    ...original,
    getFiles: vi.fn(),
    retryIngestionJob: vi.fn(),
    cancelIngestionJob: vi.fn(),
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

function makeFile(ingestion: IngestionJob | null): File {
  return {
    id: "file-1",
    is_processed: false,
    last_opened_at: null,
    deletion_state: "active",
    deletion_error: null,
    deletion_attempts: 0,
    ingestion,
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
