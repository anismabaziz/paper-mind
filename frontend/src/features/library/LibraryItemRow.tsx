import { cn } from "@/lib/utils";
import { displayTitle, indexStatusLine } from "@/types/db";
import type { File as DbFile } from "@/types/db";
import RowStateBadge from "./LibraryItemBadge";
import RowAction, { type LibraryItemActions } from "./LibraryItemAction";
import { jobStatusLine, type LibraryItemStatus } from "./library-item-status";

type Props = {
  item: DbFile;
  index: number;
  active: boolean;
  status: LibraryItemStatus;
  /** Refused while the Document is mid-job: its bytes are readable, but its
   * passages are not there yet, so opening it would show an empty reader. */
  onOpen: () => void;
} & LibraryItemActions;

/**
 * One Document in the rail: what it is, what is happening to it, and the one
 * action that is worth offering right now.
 */
export default function LibraryItemRow({ item, index, active, status, onOpen, ...actions }: Props) {
  const { job, isDeleting, isDeleteFailed, isStaleIndex, isReindexing, isJobActive } = status;

  return (
    <li className="min-w-0 max-w-full overflow-hidden" data-testid={`library-item-${item.name}`}>
      <div
        className={cn(
          "group relative flex min-w-0 max-w-full items-center gap-0 overflow-hidden border-l-2 text-left transition-colors",
          active ? "border-marker bg-paper" : "border-transparent hover:border-rule hover:bg-paper/70",
          (isDeleting || actions.deletePending) && "opacity-50 pointer-events-none",
        )}
      >
        <button
          type="button"
          onClick={onOpen}
          aria-disabled={isJobActive}
          aria-label={isJobActive ? `${displayTitle(item)}, indexing` : undefined}
          className={cn(
            "flex min-w-0 max-w-full flex-1 flex-col gap-1 overflow-hidden px-3 py-3 text-left",
            isJobActive && "cursor-default",
          )}
        >
          <span className="flex w-full min-w-0 max-w-full items-center justify-between overflow-hidden font-mono text-[0.58rem] text-ink-faint">
            <span>0{index + 1}</span>
            <RowStateBadge item={item} status={status} />
          </span>

          <span
            className={cn(
              "block min-w-0 max-w-full overflow-hidden text-[0.78rem] leading-snug break-words line-clamp-2",
              active ? "font-medium text-ink" : "text-ink-soft group-hover:text-ink",
            )}
          >
            {displayTitle(item)}
          </span>
          <span className="block min-w-0 max-w-full truncate overflow-hidden text-[0.65rem] text-ink-faint">
            {item.metadata.content_type.split("/").pop()?.toUpperCase() ?? "PDF"} ·{" "}
            {isDeleting
              ? "Deleting"
              : isDeleteFailed
                ? (item.deletion_error ?? "Delete failed — retry")
                : isStaleIndex || isReindexing
                  ? indexStatusLine(item.index)
                  : jobStatusLine(job)}
          </span>
        </button>

        <div className="pr-1">
          <RowAction item={item} status={status} active={active} {...actions} />
        </div>
      </div>
    </li>
  );
}