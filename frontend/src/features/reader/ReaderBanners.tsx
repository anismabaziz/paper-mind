import { FailureNotice } from "@/components/FailureNotice";

type Props = {
  retryFailed: boolean;
  retryDetail: string | null;
  cancelFailed: boolean;
  cancelDetail: string | null;
  statusStale: boolean;
  statusMissing: boolean;
  statusErrorText: string | null;
  deleteFailed: boolean;
  deleteDetail: string | null;
  onRetryStatus: () => void;
  onRetryDelete: () => void;
  onDismissDelete: () => void;
};

/**
 * What is wrong above the sheet, said in the reader's own terms.
 *
 * A status check that failed with a previous answer still on screen shows that
 * answer marked as stale, rather than pretending the Document's state is
 * unknown — the reader can still read, they just cannot trust the label.
 */
export default function ReaderBanners({
  retryFailed,
  retryDetail,
  cancelFailed,
  cancelDetail,
  statusStale,
  statusMissing,
  statusErrorText,
  deleteFailed,
  deleteDetail,
  onRetryStatus,
  onRetryDelete,
  onDismissDelete,
}: Props) {
  return (
    <>
      {retryFailed && (
        <Alert heading="Retry failed" detail={retryDetail ?? "The document could not be queued for indexing again."} />
      )}
      {cancelFailed && (
        <Alert heading="Cancellation failed" detail={cancelDetail ?? "The indexing job could not be cancelled."} />
      )}
      {statusStale && (
        <div className="border-b border-rule bg-background px-5 py-2">
          <FailureNotice
            testId="reader-status-stale"
            variant="stale"
            title="Showing the last confirmed state"
            message="This document's indexing status could not be refreshed. Nothing changed on the server."
            actionLabel="Refresh status"
            onAction={onRetryStatus}
          />
        </div>
      )}
      {statusMissing && (
        <div className="border-b border-destructive/40 bg-destructive/5 px-5 py-2">
          <FailureNotice
            testId="reader-status-error"
            title="Could not check indexing status"
            message={`Questions are paused, not indexing — no confirmed state is available. ${statusErrorText ?? "The status request failed."}`}
            actionLabel="Retry status check"
            onAction={onRetryStatus}
          />
        </div>
      )}
      {deleteFailed && (
        <div role="alert" className="border-b border-destructive/40 bg-destructive/5 px-5 py-2">
          <p className="text-xs font-medium text-destructive">
            Delete failed — the document was kept.
          </p>
          <p className="mt-0.5 text-[0.65rem] text-ink-soft">
            {deleteDetail ?? "Retry deletion."}
          </p>
          <div className="mt-1.5 flex gap-2">
            <button
              type="button"
              onClick={onRetryDelete}
              className="border border-ink bg-ink px-2 py-1 font-mono text-[0.6rem] text-paper hover:bg-ink/90"
            >
              Retry delete
            </button>
            <button
              type="button"
              onClick={onDismissDelete}
              className="border border-rule bg-paper px-2 py-1 font-mono text-[0.6rem] hover:border-ink"
            >
              Dismiss
            </button>
          </div>
        </div>
      )}
    </>
  );
}

function Alert({ heading, detail }: { heading: string; detail: string }) {
  return (
    <div role="alert" className="border-b border-destructive/40 bg-destructive/5 px-5 py-2">
      <p className="text-xs font-medium text-destructive">{heading}</p>
      <p className="mt-0.5 text-[0.65rem] text-ink-soft">{detail}</p>
    </div>
  );
}