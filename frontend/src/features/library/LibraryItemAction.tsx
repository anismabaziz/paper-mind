import { Loader2, MoreHorizontal, RefreshCw, RotateCw, Square, Trash2 } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu";
import { displayTitle, ingestionStageLabel, isIngestionCancellable } from "@/types/db";
import type { File as DbFile } from "@/types/db";
import type { LibraryItemStatus } from "./library-item-status";

export type LibraryItemActions = {
  /** Whether a removal is running at all, for the buttons to wait behind. */
  deletePending: boolean;
  onRetryDelete: () => void;
  onReindex: () => void;
  onRetryIndexing: () => void;
  onCancelIndexing: () => void;
  onRemove: () => void;
};

type Props = {
  item: DbFile;
  status: LibraryItemStatus;
  active: boolean;
} & LibraryItemActions;

/**
 * The single control a row offers: the one action its current state calls for.
 *
 * Offering everything at once would invite a reader to reindex a Document that
 * is still indexing, or to remove one whose removal has not finished. What is
 * actionable depends on the state, so exactly one of these is on screen.
 */
export default function RowAction({
  item,
  status,
  active,
  deletePending,
  onRetryDelete,
  onReindex,
  onRetryIndexing,
  onCancelIndexing,
  onRemove,
}: Props) {
  const {
    job,
    isDeleting,
    isDeleteFailed,
    isStaleIndex,
    isReindexing,
    isJobActive,
    isJobRetryable,
    isRetrying,
    isCancelling,
  } = status;

  if (isDeleteFailed) {
    return (
      <button
        type="button"
        onClick={onRetryDelete}
        disabled={deletePending}
        title={item.deletion_error ?? "Retry deletion"}
        className="mr-1 border border-destructive/50 bg-paper px-2 py-1 font-mono text-[0.6rem] text-destructive hover:border-destructive disabled:opacity-40"
      >
        Retry
      </button>
    );
  }
  if (isStaleIndex || isReindexing) {
    return (
      <button
        type="button"
        onClick={onReindex}
        disabled={isReindexing}
        title={item.index?.change_details.map((c) => c.label).join(", ") ?? "Reindex"}
        aria-label={`Reindex ${displayTitle(item)}`}
        data-testid={`reindex-${item.name}`}
        className="mr-1 inline-flex items-center gap-1 border border-destructive/50 bg-paper px-2 py-1 font-mono text-[0.6rem] text-destructive hover:border-destructive disabled:opacity-40"
      >
        {isReindexing ? <Loader2 className="size-3 animate-spin" /> : <RefreshCw className="size-3" />}
        Reindex
      </button>
    );
  }
  if (isJobRetryable) {
    return (
      <button
        type="button"
        onClick={onRetryIndexing}
        disabled={isRetrying}
        title={job?.error_message ?? "Retry indexing"}
        aria-label={`Retry indexing ${displayTitle(item)}`}
        data-testid={`retry-${item.name}`}
        className="mr-1 inline-flex items-center gap-1 border border-destructive/50 bg-paper px-2 py-1 font-mono text-[0.6rem] text-destructive hover:border-destructive disabled:opacity-40"
      >
        {isRetrying ? <Loader2 className="size-3 animate-spin" /> : <RotateCw className="size-3" />}
        Retry
      </button>
    );
  }
  if (isJobActive) {
    return (
      <>
        <span
          className="mr-1 grid size-7 place-items-center"
          role="status"
          aria-label={`${ingestionStageLabel(job?.stage ?? "queued")} ${job?.progress ?? 0}%`}
        >
          <Loader2 className="size-3 animate-spin" />
        </span>
        {isIngestionCancellable(job?.state) && (
          <button
            type="button"
            onClick={onCancelIndexing}
            disabled={isCancelling}
            title="Cancel indexing"
            aria-label={`Cancel indexing ${displayTitle(item)}`}
            className="mr-1 grid size-7 place-items-center border border-rule bg-paper text-ink-soft hover:border-destructive hover:text-destructive disabled:opacity-40"
          >
            {isCancelling ? <Loader2 className="size-3 animate-spin" /> : <Square className="size-3" />}
          </button>
        )}
      </>
    );
  }
  if (isDeleting) {
    return (
      <span className="grid size-7 place-items-center text-ink-faint">
        <Loader2 className="size-3.5 animate-spin" />
      </span>
    );
  }
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          data-testid={`document-menu-${item.name}`}
          type="button"
          aria-label={`Actions for ${displayTitle(item)}`}
          aria-haspopup="menu"
          className={cn(
            "grid size-7 place-items-center border border-transparent text-ink-faint hover:border-rule hover:bg-paper hover:text-ink",
            active ? "opacity-100" : "opacity-0 group-hover:opacity-100",
          )}
        >
          <MoreHorizontal className="size-3.5" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-[140px] border-rule bg-paper p-1">
        <DropdownMenuItem
          className="flex items-center gap-2 text-xs text-destructive focus:bg-destructive/10 focus:text-destructive"
          onClick={onRemove}
          data-testid={`delete-paper-${item.name}`}
        >
          <Trash2 className="size-3.5" /> Remove Paper
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}