import { Suspense } from "react";
import { RotateCw } from "lucide-react";
import { FailureNotice } from "@/components/FailureNotice";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import IndexingOverlay from "./IndexingOverlay";
import type { IngestionJob } from "@/types/db";

type Props = {
  page: number;
  downloadError: string | null;
  onRetryDownload: () => void;
  onChooseAnother: () => void;
  isProcessed: boolean;
  isJobActive: boolean;
  isJobRetryable: boolean;
  isJobFailed: boolean;
  canCancel: boolean;
  ingestionJob: IngestionJob | null;
  retryPending: boolean;
  cancelPending: boolean;
  onRetryIndexing: () => void;
  onCancelIndexing: () => void;
  children: React.ReactNode;
};

/**
 * The white sheet the PDF renders into, and whatever is standing in front of it.
 *
 * The renderer is loaded on demand because it is the only part of the reader
 * that needs pdf.js: the rail and the chat are useful before a single page is
 * rendered, and they should not wait for a worker to start.
 */
export default function ReaderDocumentSheet({
  page,
  downloadError,
  onRetryDownload,
  onChooseAnother,
  isProcessed,
  isJobActive,
  isJobRetryable,
  isJobFailed,
  canCancel,
  ingestionJob,
  retryPending,
  cancelPending,
  onRetryIndexing,
  onCancelIndexing,
  children,
}: Props) {
  return (
    <div className="paper-grain relative bg-paper shadow-sheet">
      <div className="flex items-center justify-between border-b border-rule px-6 py-3">
        <span className="label-meta">Page {String(page).padStart(2, "0")}</span>
        <Separator className="mx-3 flex-1 bg-rule" />
        <span className="label-meta">p. {page}</span>
      </div>
      <div className="relative bg-canvas p-3">
        <div className="overflow-hidden border border-rule bg-white">
          {downloadError ? (
            <div
              className="grid min-h-[760px] place-items-center bg-white p-6 text-center"
              data-testid="reader-download-error"
            >
              <div>
                <FailureNotice
                  title="Could not download this document"
                  message={`${downloadError} The document is still in your library — this is a download failure, not a missing document.`}
                />
                <div className="mt-3 flex items-center justify-center gap-2">
                  <Button
                    type="button"
                    size="sm"
                    onClick={onRetryDownload}
                    className="gap-1.5 rounded-none border-ink bg-ink px-3 py-1.5 font-mono text-[0.65rem] text-paper hover:bg-ink/90"
                  >
                    <RotateCw className="size-3" /> Retry download
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={onChooseAnother}
                    className="gap-1.5 rounded-none border-rule bg-paper px-3 py-1.5 font-mono text-[0.65rem] text-ink-soft hover:border-ink"
                  >
                    Choose another document
                  </Button>
                </div>
              </div>
            </div>
          ) : (
            <Suspense
              fallback={
                <div className="grid h-[760px] place-items-center bg-white">
                  <p className="font-mono text-xs text-ink-faint">Loading document…</p>
                </div>
              }
            >
              {children}
            </Suspense>
          )}
        </div>
        <IndexingOverlay
          isProcessed={isProcessed}
          isJobActive={isJobActive}
          isJobRetryable={isJobRetryable}
          isJobFailed={isJobFailed}
          canCancel={canCancel}
          ingestionJob={ingestionJob}
          retryPending={retryPending}
          cancelPending={cancelPending}
          onRetry={onRetryIndexing}
          onCancel={onCancelIndexing}
        />
      </div>

      <div className="flex items-center justify-between border-t border-rule px-10 py-4">
        <span className="label-meta">p. {String(page).padStart(2, "0")}</span>
        <span className="label-meta">{String(page).padStart(2, "0")}</span>
      </div>
    </div>
  );
}
