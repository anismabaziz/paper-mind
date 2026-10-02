import { cn } from "@/lib/utils";
import { displayTitle, ingestionStageLabel } from "@/types/db";
import type { File as DbFile, IngestionJob } from "@/types/db";

type Props = {
  file: DbFile;
  page: number;
  isProcessed: boolean;
  isJobActive: boolean;
  isJobRetryable: boolean;
  isJobFailed: boolean;
  ingestionJob: IngestionJob | null;
  retryPending: boolean;
  onRetry: () => void;
};

/**
 * The first sheet in the reader: what this Document is and whether it can be
 * asked questions yet.
 *
 * The Document stays readable while it indexes — the PDF is already here — so
 * this says what is missing rather than hiding the paper behind it.
 */
export default function ReaderTitleSheet({
  file,
  page,
  isProcessed,
  isJobActive,
  isJobRetryable,
  isJobFailed,
  ingestionJob,
  retryPending,
  onRetry,
}: Props) {
  return (
    <article className="paper-grain mb-6 bg-paper px-6 sm:px-10 pt-10 pb-8 shadow-sheet">
      <p className="label-meta">Research paper · {file.metadata.content_type}</p>
      <h1 className="mt-3 font-serif text-[1.7rem] leading-[1.15] font-medium text-balance">
        {displayTitle(file)}
      </h1>
      <div className="mt-5 flex items-center gap-3 border-y border-rule py-4">
        <span
          className={cn(
            "inline-flex border px-2 py-1 font-mono text-[0.62rem] uppercase tracking-wider",
            isProcessed
              ? "border-marker bg-marker-soft text-marker"
              : "border-rule bg-canvas text-ink-faint",
          )}
        >
          {isJobActive
            ? `${ingestionStageLabel(ingestionJob?.stage ?? "queued")} · ${ingestionJob?.progress ?? 0}%`
            : isJobRetryable
              ? isJobFailed
                ? "Indexing failed"
                : "Indexing cancelled"
              : isProcessed
                ? "Ready for questions"
                : "Indexing"}
        </span>
        {isJobRetryable && isProcessed && !isJobActive && (
          <button
            type="button"
            onClick={onRetry}
            disabled={retryPending}
            className="ml-2 border border-rule bg-paper px-2 py-1 font-mono text-[0.6rem] text-ink-soft hover:border-ink hover:text-ink disabled:opacity-40"
          >
            Retry
          </button>
        )}
        <span className="font-mono text-[0.62rem] text-ink-faint">
          Page {String(page).padStart(2, "0")}
        </span>
      </div>

      <p className="mt-4 font-serif text-[0.95rem] leading-[1.7] text-ink-soft italic">
        <span className="mr-2 font-mono text-[0.62rem] tracking-[0.14em] text-marker not-italic uppercase">
          Abstract
        </span>
        This workspace keeps every answer tied to the passage it came from. Ask a question in the
        companion and the document stays open beside it — no context lost.
      </p>
    </article>
  );
}