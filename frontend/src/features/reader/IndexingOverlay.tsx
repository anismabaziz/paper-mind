import { Loader2, RotateCw, Square } from "lucide-react";
import { ingestionStageLabel } from "@/types/db";
import type { IngestionJob } from "@/types/db";

type Props = {
  isProcessed: boolean;
  isJobActive: boolean;
  isJobRetryable: boolean;
  isJobFailed: boolean;
  canCancel: boolean;
  ingestionJob: IngestionJob | null;
  retryPending: boolean;
  cancelPending: boolean;
  onRetry: () => void;
  onCancel: () => void;
};

/**
 * What stands over the pages while the Document cannot yet be searched.
 *
 * The pages stay mounted underneath rather than being replaced, so the reader
 * can scroll while the job runs and the paper does not vanish and come back.
 */
export default function IndexingOverlay({
  isProcessed,
  isJobActive,
  isJobRetryable,
  isJobFailed,
  canCancel,
  ingestionJob,
  retryPending,
  cancelPending,
  onRetry,
  onCancel,
}: Props) {
  if (isProcessed && !isJobActive) return null;

  return (
    <div className="absolute inset-3 grid place-items-center bg-paper/70 backdrop-blur-[1px]">
      <div className="rounded-sm border border-rule bg-paper px-4 py-3 text-center shadow-sheet">
        <p className="font-mono text-xs font-medium" role="status">
          {isJobFailed
            ? "Indexing failed"
            : isJobRetryable
              ? "Indexing cancelled"
              : `${ingestionStageLabel(ingestionJob?.stage ?? "queued")} · ${ingestionJob?.progress ?? 0}%`}
        </p>
        <p className="mt-1 text-xs text-ink-soft">
          {isJobFailed
            ? (ingestionJob?.error_message ?? "Indexing failed before this document was ready.")
            : isJobRetryable
              ? "The indexing job was cancelled. You can retry it when ready."
              : "Semantic vectors are being generated — chat will unlock when this pass finishes."}
        </p>
        {isJobRetryable && (
          <button
            type="button"
            onClick={onRetry}
            disabled={retryPending}
            className="mt-3 inline-flex items-center gap-1.5 border border-ink bg-ink px-3 py-1.5 font-mono text-[0.65rem] text-paper hover:bg-ink/90 disabled:opacity-40"
          >
            {retryPending ? (
              <Loader2 className="size-3 animate-spin" />
            ) : (
              <RotateCw className="size-3" />
            )}
            Retry indexing
          </button>
        )}
        {isJobActive && canCancel && (
          <button
            type="button"
            onClick={onCancel}
            disabled={cancelPending}
            className="mt-3 ml-2 inline-flex items-center gap-1.5 border border-rule bg-paper px-3 py-1.5 font-mono text-[0.65rem] text-ink-soft hover:border-destructive hover:text-destructive disabled:opacity-40"
          >
            {cancelPending ? (
              <Loader2 className="size-3 animate-spin" />
            ) : (
              <Square className="size-3" />
            )}
            Cancel indexing
          </button>
        )}
      </div>
    </div>
  );
}