import { useFileStatus, useFileMeta, useDeleteFile, useRetryIngestion, useCancelIngestion } from "./useFiles";
import { getErrorMessage } from "@/lib/api-error";
import { readerStatusLine } from "@/features/reader/reader-status";
import {
  isIngestionActive,
  isIngestionCancellable,
  isIngestionRetryable,
} from "@/types/db";
import type { File as DbFile } from "@/types/db";

/**
 * What the open Document allows right now: whether it is searchable, what its
 * indexing job is doing, and the three actions that job can be given.
 *
 * These are read together because they are one decision. Whether chat can run,
 * whether the reader shows an overlay, and which button appears all follow from
 * the same two facts, and reading them from three places is how they drift
 * apart and one of them ends up promising something the others refuse.
 */
export function useReaderDocumentState(
  file: DbFile | null,
  onDeselect: () => void,
) {
  const statusQuery = useFileStatus(file);
  const metaQuery = useFileMeta(file);

  const deleteMutation = useDeleteFile({
    onSuccess: (_data, variables) => {
      if (file?.id === variables.id) onDeselect();
    },
  });
  const retryMutation = useRetryIngestion();
  const cancelMutation = useCancelIngestion();

  const ingestionJob = statusQuery.data?.ingestion ?? null;
  const isProcessed = statusQuery.data?.is_processed ?? false;
  const isJobActive = isIngestionActive(ingestionJob?.state);
  const isJobFailed = ingestionJob?.state === "failed";
  const isJobRetryable = isIngestionRetryable(ingestionJob?.state);

  return {
    file,
    ingestionJob,
    isProcessed,
    isJobActive,
    isJobFailed,
    isJobRetryable,
    canCancel: isIngestionCancellable(ingestionJob?.state),
    statusLine: readerStatusLine({
      file,
      ingestionJob,
      isProcessed,
      isJobActive,
      isJobRetryable,
      isJobFailed,
    }),
    /** A failed status check with an earlier answer still counts as stale. */
    statusStale: statusQuery.isError && Boolean(statusQuery.data),
    statusMissing: statusQuery.isError && !statusQuery.data,
    statusErrorText: getErrorMessage(statusQuery.error) || null,
    refetchStatus: () => void statusQuery.refetch(),
    outline: metaQuery.data?.outline ?? [],
    outlinePageCount: metaQuery.data?.pageCount ?? null,
    outlineState: metaQuery.isError ? ("error" as const) : metaQuery.isLoading ? ("loading" as const) : ("ready" as const),
    outlineErrorText: getErrorMessage(metaQuery.error) || null,
    refetchOutline: () => void metaQuery.refetch(),
    retry: {
      pending: retryMutation.isPending,
      failed: retryMutation.isError,
      detail: getErrorMessage(retryMutation.error) || null,
      run: () => file && retryMutation.mutate(file.name),
    },
    cancel: {
      pending: cancelMutation.isPending,
      failed: cancelMutation.isError,
      detail: getErrorMessage(cancelMutation.error) || null,
      run: () => file && cancelMutation.mutate(file.name),
    },
    remove: {
      pending: deleteMutation.isPending,
      failed: deleteMutation.isError,
      detail: getErrorMessage(deleteMutation.error) || null,
      run: () => file && deleteMutation.mutate(file),
      dismiss: () => deleteMutation.reset(),
    },
  };
}