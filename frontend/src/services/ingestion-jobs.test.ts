import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { IngestionJob } from "@/types/db";

const queuedJob: IngestionJob = {
  id: "job-1",
  file_id: "file-1",
  filename: "doc.pdf",
  generation: 1,
  state: "queued",
  stage: "queued",
  progress: 0,
  attempt: 1,
  error_category: null,
  error_message: null,
  worker_id: null,
  created_at: "2026-09-24T10:00:00+00:00",
  updated_at: "2026-09-24T10:00:00+00:00",
  started_at: null,
  finished_at: null,
  heartbeat_at: null,
};

const request = vi.fn();

vi.mock("./client", () => ({
  default: {
    get: (...args: unknown[]) => request("get", ...args),
    post: (...args: unknown[]) => request("post", ...args),
  },
  apiBaseUrl: "http://127.0.0.1:3000",
}));

import { retryIngestionJob, uploadFile } from "./files";

describe("ingestion job routes", () => {
  beforeEach(() => {
    request.mockReset();
  });

  afterEach(() => {
    request.mockReset();
  });

  it("retries a failed job through the retry route", async () => {
    const retried = { ...queuedJob, attempt: 2 };
    request.mockResolvedValue({ data: { job: retried } });

    await expect(retryIngestionJob("doc.pdf")).resolves.toEqual(retried);
    expect(request).toHaveBeenCalledWith("post", "/ingestion-jobs/doc.pdf/retry");
  });

  it("escapes the document name in the retry route", async () => {
    request.mockResolvedValue({ data: { job: queuedJob } });

    await retryIngestionJob("a b/c.pdf");

    expect(request).toHaveBeenCalledWith(
      "post",
      "/ingestion-jobs/a%20b%2Fc.pdf/retry"
    );
  });

  it("returns the queued job with the upload response", async () => {
    request.mockResolvedValue({
      data: {
        message: "File uploaded successfully",
        file: { id: "file-1", name: "doc.pdf" },
        job: queuedJob,
      },
    });

    const bytes = new Uint8Array([0x25, 0x50, 0x44, 0x46]);
    const result = await uploadFile(new File([bytes], "doc.pdf"));

    expect(result.job).toEqual(queuedJob);
  });
});
