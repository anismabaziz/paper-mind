import { ingestionStageLabel } from "@/types/db";
import type { File as DbFile, IngestionJob } from "@/types/db";

/**
 * What the reader's title line says about the Document's indexing, in one
 * place: the toolbar, the title sheet and the overlay would otherwise each
 * spell the same state out slightly differently.
 */
export function readerStatusLine({
  file,
  ingestionJob,
  isProcessed,
  isJobActive,
  isJobRetryable,
  isJobFailed,
}: {
  file: DbFile | null;
  ingestionJob: IngestionJob | null;
  isProcessed: boolean;
  isJobActive: boolean;
  isJobRetryable: boolean;
  isJobFailed: boolean;
}): string {
  if (!file) return "No document selected";
  if (file.deletion_state === "deleting") return "Deleting";
  if (file.deletion_state === "delete_failed") return "Delete failed";
  if (isJobActive) {
    return `${ingestionStageLabel(ingestionJob?.stage ?? "queued")} ${ingestionJob?.progress ?? 0}%`;
  }
  if (isJobRetryable) return isJobFailed ? "Indexing failed" : "Indexing cancelled";
  return isProcessed ? "Indexed" : "Indexing";
}
