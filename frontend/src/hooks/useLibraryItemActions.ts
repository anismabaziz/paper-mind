import { useDeleteFile, useRetryIngestion, useCancelIngestion, useReindex } from "./useFiles";
import { getErrorMessage } from "@/lib/api-error";
import { libraryItemStatus } from "@/features/library/library-item-status";
import type { LibraryFailure } from "@/features/library/LibraryFailures";
import type { LibraryRow } from "@/features/library/LibraryList";
import type { File as DbFile } from "@/types/db";

/**
 * The four things a reader can do to a Document after it is in the library:
 * remove it, retry its indexing, cancel it, or reindex it.
 *
 * Each is a single in-flight request, so each row has to be able to say "this
 * one is already being retried" rather than offering the same action twice.
 * That state is what `rowsFor` decides, and it is decided once here instead of
 * per row.
 */
export function useLibraryItemActions({
  selectedFileId,
  onSelect,
  onDeselect,
}: {
  selectedFileId: string | null;
  onSelect: (file: DbFile) => void;
  onDeselect: () => void;
}) {
  const deleteMutation = useDeleteFile({
    onSuccess: (_data, variables) => {
      if (selectedFileId === variables.id) onDeselect();
    },
  });
  const retryMutation = useRetryIngestion();
  const cancelMutation = useCancelIngestion();
  const reindexMutation = useReindex();

  const removingId = deleteMutation.isPending
    ? (deleteMutation.variables as DbFile | undefined)?.id ?? null
    : null;
  const retryingName = retryMutation.isPending
    ? (retryMutation.variables as string | undefined) ?? null
    : null;
  const cancellingName = cancelMutation.isPending
    ? (cancelMutation.variables as string | undefined) ?? null
    : null;
  const reindexingName = reindexMutation.isPending
    ? (reindexMutation.variables as string | undefined) ?? null
    : null;

  const rowsFor = (files: DbFile[]): LibraryRow[] =>
    files.map((item, index) => {
      const status = libraryItemStatus(item, {
        removingId,
        retryingName,
        cancellingName,
        reindexingName,
      });
      return {
        item,
        index,
        active: selectedFileId === item.id,
        status,
        deletePending: deleteMutation.isPending,
        onOpen: () => {
          // A failed job still has readable PDF bytes, so the reader stays
          // reachable to inspect and retry it.
          if (!status.isJobActive) onSelect(item);
        },
        onRemove: () => deleteMutation.mutate(item),
        onRetryDelete: () => deleteMutation.mutate(item),
        onReindex: () => reindexMutation.mutate(item.name),
        onRetryIndexing: () => retryMutation.mutate(item.name),
        onCancelIndexing: () => cancelMutation.mutate(item.name),
      };
    });

  const failures = ([
    { key: "retry", heading: "Retry failed", detail: getErrorMessage(retryMutation.error), reset: () => retryMutation.reset() },
    { key: "cancel", heading: "Cancellation failed", detail: getErrorMessage(cancelMutation.error), reset: () => cancelMutation.reset() },
    { key: "reindex", heading: "Reindex failed", detail: getErrorMessage(reindexMutation.error), reset: () => reindexMutation.reset() },
  ] satisfies LibraryFailure[]).filter((failure) => Boolean(failure.detail));

  return {
    rowsFor,
    failures,
    deleteFailure: deleteMutation.isError
      ? {
          target: deleteMutation.variables as DbFile | undefined,
          detail: getErrorMessage(deleteMutation.error) || null,
          pending: deleteMutation.isPending,
          onRetry: () => {
            const target = deleteMutation.variables as DbFile | undefined;
            if (target) deleteMutation.mutate(target);
          },
          onDismiss: () => deleteMutation.reset(),
        }
      : null,
  };
}