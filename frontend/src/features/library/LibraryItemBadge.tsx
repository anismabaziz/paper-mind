import { AlertTriangle, Loader2 } from "lucide-react";
import { formatFileSize } from "@/lib/format";
import { ingestionStageLabel } from "@/types/db";
import type { File as DbFile } from "@/types/db";
import type { LibraryItemStatus } from "./library-item-status";

/**
 * The one word at the top right of a row: the state a reader would act on.
 *
 * A healthy Document shows its size instead, because there is nothing about it
 * a reader needs to do.
 */
export default function RowStateBadge({ item, status }: { item: DbFile; status: LibraryItemStatus }) {
  const {
    job,
    isDeleting,
    isDeleteFailed,
    isStaleIndex,
    isReindexing,
    isJobActive,
    isJobFailed,
    isJobRetryable,
    isRetrying,
    isCancelling,
  } = status;

  if (isDeleting) {
    return (
      <span className="inline-flex items-center gap-1">
        <Loader2 className="size-2.5 animate-spin" />
        Deleting
      </span>
    );
  }
  if (isDeleteFailed) return <span className="text-destructive">Delete failed</span>;
  if (isStaleIndex || isReindexing) {
    return (
      <span className="inline-flex items-center gap-1 text-destructive">
        {isReindexing ? <Loader2 className="size-2.5 animate-spin" /> : <AlertTriangle className="size-2.5" />}
        {isReindexing ? "Reindexing" : "Stale"}
      </span>
    );
  }
  if (isJobFailed) {
    return (
      <span className="inline-flex items-center gap-1 text-destructive">
        <AlertTriangle className="size-2.5" />
        Failed
      </span>
    );
  }
  if (isJobActive) {
    return (
      <span className="inline-flex items-center gap-1">
        <Loader2 className="size-2.5 animate-spin" />
        {isCancelling
          ? "Cancelling"
          : isRetrying
            ? "Retrying"
            : ingestionStageLabel(job?.stage ?? "queued")}
      </span>
    );
  }
  if (isJobRetryable) {
    return (
      <span className="inline-flex items-center gap-1 text-destructive">
        <AlertTriangle className="size-2.5" />
        Retry
      </span>
    );
  }
  return <span>{formatFileSize(item.metadata.size)}</span>;
}
