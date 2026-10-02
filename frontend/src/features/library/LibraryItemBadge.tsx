import { AlertTriangle, Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
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
      <Badge variant="secondary" className="gap-1 font-mono text-[0.6rem] font-normal">
        <Loader2 className="size-2.5 animate-spin" aria-hidden="true" />
        Deleting
      </Badge>
    );
  }
  if (isDeleteFailed)
    return (
      <Badge variant="destructive" className="font-mono text-[0.6rem] font-normal">
        Delete failed
      </Badge>
    );
  if (isStaleIndex || isReindexing) {
    return (
      <Badge variant="destructive" className="gap-1 font-mono text-[0.6rem] font-normal">
        {isReindexing ? (
          <Loader2 className="size-2.5 animate-spin" aria-hidden="true" />
        ) : (
          <AlertTriangle className="size-2.5" aria-hidden="true" />
        )}
        {isReindexing ? "Reindexing" : "Stale"}
      </Badge>
    );
  }
  if (isJobFailed) {
    return (
      <Badge variant="destructive" className="gap-1 font-mono text-[0.6rem] font-normal">
        <AlertTriangle className="size-2.5" aria-hidden="true" />
        Failed
      </Badge>
    );
  }
  if (isJobActive) {
    return (
      <Badge variant="secondary" className="gap-1 font-mono text-[0.6rem] font-normal">
        <Loader2 className="size-2.5 animate-spin" aria-hidden="true" />
        {isCancelling
          ? "Cancelling"
          : isRetrying
            ? "Retrying"
            : ingestionStageLabel(job?.stage ?? "queued")}
      </Badge>
    );
  }
  if (isJobRetryable) {
    return (
      <Badge variant="destructive" className="gap-1 font-mono text-[0.6rem] font-normal">
        <AlertTriangle className="size-2.5" aria-hidden="true" />
        Retry
      </Badge>
    );
  }
  return <span>{formatFileSize(item.metadata.size)}</span>;
}
