import { File, Upload } from "lucide-react";
import { FailureNotice } from "@/components/FailureNotice";
import LibraryItemRow from "./LibraryItemRow";
import type { LibraryItemActions } from "./LibraryItemAction";
import type { LibraryItemStatus } from "./library-item-status";
import type { File as DbFile } from "@/types/db";

/** Everything one row needs, decided by the rail rather than repeated here. */
export type LibraryRow = LibraryItemActions & {
  item: DbFile;
  index: number;
  active: boolean;
  status: LibraryItemStatus;
  onOpen: () => void;
};

type Props = {
  rows: LibraryRow[];
  tab: "library" | "recent";
  /** Which of the four states the listing is in, before it has anything to show. */
  state: "pending" | "error" | "empty" | "ready";
  errorText: string | null;
  onRetryLoad: () => void;
  onUpload: () => void;
  onShowLibrary: () => void;
};

/**
 * The rail's listing: still loading, failed, empty, or the Documents themselves.
 *
 * Each of those is a different thing to say, and the failed one in particular is
 * not an empty library — conflating them would have a reader conclude their
 * papers were gone.
 */
export default function LibraryList({
  rows,
  tab,
  state,
  errorText,
  onRetryLoad,
  onUpload,
  onShowLibrary,
}: Props) {
  if (state === "pending") {
    return (
      <div className="space-y-2 px-1">
        {[...Array(3)].map((_, i) => (
          <div key={i} className="h-[76px] w-full animate-pulse rounded-sm border border-rule bg-paper/60" />
        ))}
      </div>
    );
  }

  if (state === "error") {
    return (
      <FailureNotice
        testId="library-error"
        title="Could not load your library"
        message={`${errorText ?? "The library request failed."} Nothing was deleted — this is a loading failure, not an empty library.`}
        actionLabel="Retry library load"
        onAction={onRetryLoad}
      />
    );
  }

  if (state === "empty") {
    return (
      <div
        className="mx-1 rounded-sm border border-dashed border-rule bg-paper/40 px-4 py-8 text-center"
        data-testid="library-empty"
      >
        <div className="mx-auto grid size-8 place-items-center border border-rule bg-paper text-ink-faint">
          <File className="size-3.5" />
        </div>
        {tab === "recent" ? (
          <>
            <p className="mt-3 text-xs font-medium">No recent readings</p>
            <p className="mt-1 text-[0.65rem] leading-relaxed text-ink-faint">
              Open a paper and it will show up here.
            </p>
            <button
              type="button"
              onClick={onShowLibrary}
              className="mt-4 inline-flex items-center gap-1.5 border border-rule bg-paper px-3 py-1.5 font-mono text-[0.65rem] hover:border-ink"
            >
              Back to library
            </button>
          </>
        ) : (
          <>
            <p className="mt-3 text-xs font-medium">No documents</p>
            <p className="mt-1 text-[0.65rem] leading-relaxed text-ink-faint">
              Ingest a PDF to begin analysis.
            </p>
            <button
              type="button"
              onClick={onUpload}
              className="mt-4 inline-flex items-center gap-1.5 border border-ink bg-ink px-3 py-1.5 font-mono text-[0.65rem] text-paper hover:bg-ink/90"
            >
              <Upload className="size-3" /> Ingest Document
            </button>
          </>
        )}
      </div>
    );
  }

  return (
    <ul className="min-w-0 max-w-full space-y-px overflow-hidden" data-testid="library-list">
      {rows.map((row) => (
        <LibraryItemRow key={row.item.id} {...row} />
      ))}
    </ul>
  );
}