import type { File as DbFile, IngestionJob } from "@/types/db";
import {
  ingestionStageLabel,
  isIndexStale,
  isIngestionActive,
  isIngestionRetryable,
} from "@/types/db";

/** Which of the rail's own actions are in flight, and on which Document. */
export type LibraryActions = {
  removingId: string | null;
  retryingName: string | null;
  cancellingName: string | null;
  reindexingName: string | null;
};

/**
 * What one Document is doing, in the words the row shows and the buttons offer.
 *
 * A Document can be mid-way through several of these at once — a reindex that
 * was queued after a failed job, a delete that is still removing the bytes —
 * and each one changes both the label and which single action is worth
 * offering. Deriving all of it here keeps the row itself to markup.
 */
export type LibraryItemStatus = {
  job: IngestionJob | null | undefined;
  isDeleting: boolean;
  isDeleteFailed: boolean;
  isStaleIndex: boolean;
  isReindexing: boolean;
  isJobActive: boolean;
  isJobFailed: boolean;
  isJobRetryable: boolean;
  isRetrying: boolean;
  isCancelling: boolean;
};

export function libraryItemStatus(item: DbFile, actions: LibraryActions): LibraryItemStatus {
  const job = item.ingestion;
  const isRetrying = actions.retryingName === item.name;
  const isCancelling = actions.cancellingName === item.name;
  const isReindexing = actions.reindexingName === item.name;
  const isJobActive = isIngestionActive(job?.state) || isRetrying || isCancelling;
  const isStaleIndex = isIndexStale(item.index) && !isJobActive && !isReindexing;

  return {
    job,
    isDeleting: item.deletion_state === "deleting" || actions.removingId === item.id,
    isDeleteFailed: item.deletion_state === "delete_failed",
    isStaleIndex,
    isReindexing,
    isJobActive,
    isJobFailed: job?.state === "failed" && !isRetrying,
    // A row already retrying offers nothing: the button it would show is the
    // one being pressed.
    isJobRetryable: isIngestionRetryable(job?.state) && !isRetrying,
    isRetrying,
    isCancelling,
  };
}

export function jobStatusLine(job: IngestionJob | null | undefined): string {
  if (!job) return "Not indexed";
  if (job.state === "failed") {
    return job.error_message ?? "Indexing failed — retry";
  }
  if (job.state === "cancelled") return "Cancelled · retry available";
  if (job.state === "cancelling") return "Cancelling";
  if (job.state === "stale") return "Superseded by a newer attempt";
  if (job.state === "ready") return "Indexed";
  return `${ingestionStageLabel(job.stage)} · ${job.progress}%`;
}