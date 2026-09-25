import { useEffect, useMemo, useRef, useState } from "react";
import { Archive, Clock3, Search, Upload, File, Trash2, MoreHorizontal, Loader2, Settings, AlertTriangle, RotateCw, RefreshCw, Square } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { useFiles, useUploadFile, useDeleteFile, useRetryIngestion, useCancelIngestion, useReindex, useTouchFileOpened } from "@/hooks/useFiles";
import { formatFileSize } from "@/lib/format";
import usePdfStore from "@/store/pdf-state";
import useSettingsUi from "@/store/settings-ui";
import useMobileUi from "@/store/mobile-ui";
import { cn } from "@/lib/utils";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu";
import type { File as DbFile, IngestionJob } from "@/types/db";
import { displayTitle, indexStatusLine, ingestionStageLabel, isIndexStale, isIngestionActive, isIngestionCancellable, isIngestionRetryable } from "@/types/db";

function pad(n: number) {
  return String(n).padStart(2, "0");
}

const MAX_RECENTS = 12;

function jobStatusLine(job: IngestionJob | null | undefined): string {
  if (!job) return "Not indexed";
  if (job.state === "failed") {
    return job.error_message ?? "Indexing failed \u2014 retry";
  }
  if (job.state === "cancelled") return "Cancelled \u00b7 retry available";
  if (job.state === "cancelling") return "Cancelling";
  if (job.state === "stale") return "Superseded by a newer attempt";
  if (job.state === "ready") return "Indexed";
  return `${ingestionStageLabel(job.stage)} \u00b7 ${job.progress}%`;
}

export function LibraryRail() {
  const queryClient = useQueryClient();
  const { file: selectedFile, setFile } = usePdfStore();
  const [query, setQuery] = useState("");
  const [tab, setTab] = useState<"library" | "recent">("library");
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const openSettings = useSettingsUi((s) => s.open);
  const setLibraryOpen = useMobileUi((s) => s.setLibraryOpen);
  const filesQuery = useFiles();
  const touchOpened = useTouchFileOpened();

  const files = filesQuery.data?.files ?? [];

  const recents = useMemo(() => {
    const opened = files.filter((f): f is DbFile & { last_opened_at: string } => f.last_opened_at != null);
    return opened
      .sort((a, b) => b.last_opened_at.localeCompare(a.last_opened_at))
      .slice(0, MAX_RECENTS);
  }, [files]);

  function selectFile(item: DbFile) {
    setFile(item);
    touchOpened.mutate(item);
    setLibraryOpen(false);
  }

  useEffect(() => {
    if (!files.length) {
      if (selectedFile) setFile(null);
      return;
    }
    const exists = selectedFile && files.find((f) => f.id === selectedFile.id);
    if (!exists) {
      const next = files.find((f) => f.is_processed) ?? files[0];
      setFile(next);
    }
    // No touch here: this fallback is system-initiated, not the user
    // opening a paper, so it must not pollute the recent-readings order.
  }, [files, selectedFile, setFile]);

  const viewFiles: DbFile[] = tab === "recent" ? recents : files;

  const filtered = useMemo(() => {
    const v = query.trim().toLowerCase();
    if (!v) return viewFiles;
    return viewFiles.filter((f) => displayTitle(f).toLowerCase().includes(v));
  }, [viewFiles, query]);

  const uploadMutation = useUploadFile({
    onSuccess: (data) => {
      // A fresh upload counts as opened so it enters Recent readings. The
      // upload response already carries the queued ingestion job, so the
      // listing polls it directly with no client-side processing call.
      touchOpened.mutate(data.file);
      queryClient.invalidateQueries({ queryKey: ["files"] });
    },
    onError: (err) => {
      setUploadError(err instanceof Error ? err.message : String(err));
    },
  });

  const deleteMutation = useDeleteFile({
    onSuccess: (_data, variables) => {
      if (selectedFile?.id === variables.id) setFile(null);
    },
  });
  const deleteError = deleteMutation.error instanceof Error ? deleteMutation.error.message : null;
  const deleteTarget = deleteMutation.variables as DbFile | undefined;

  const retryMutation = useRetryIngestion();
  const retryTarget = retryMutation.variables as string | undefined;
  const retryError = retryMutation.error instanceof Error ? retryMutation.error.message : null;
  const cancelMutation = useCancelIngestion();
  const cancelTarget = cancelMutation.variables as string | undefined;
  const cancelError = cancelMutation.error instanceof Error ? cancelMutation.error.message : null;
  const reindexMutation = useReindex();
  const reindexTarget = reindexMutation.variables as string | undefined;
  const reindexError = reindexMutation.error instanceof Error ? reindexMutation.error.message : null;

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files?.[0]) uploadMutation.mutate(e.target.files[0]);
    if (e.target) e.target.value = "";
  }

  const listTitle = tab === "recent" ? `Recent · ${filtered.length}` : `Papers · ${filtered.length}`;

  const emptyCopy =
    tab === "recent"
      ? { title: "No recent readings", hint: "Open a paper and it will show up here." }
      : { title: "No documents", hint: "Ingest a PDF to begin analysis." };

  return (
    <aside className="flex w-64 max-w-full min-w-0 shrink-0 flex-col overflow-x-hidden border-r border-rule bg-sidebar">
      <header className="flex h-16 items-center justify-between border-b border-rule px-5">
        <div className="flex items-center gap-3">
          <span className="grid size-8 place-items-center bg-ink font-mono text-[0.7rem] font-bold text-paper">PM</span>
          <div>
            <p className="font-mono text-sm font-bold">PaperMind</p>
            <p className="text-[0.65rem] text-ink-faint">Research workspace</p>
          </div>
        </div>
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={uploadMutation.isPending}
          className="grid size-8 place-items-center border border-rule text-ink-soft hover:border-marker hover:text-marker disabled:opacity-40"
          aria-label="Upload paper"
          title="Upload paper"
        >
          {uploadMutation.isPending ? <Loader2 className="size-3.5 animate-spin" /> : <Upload className="size-3.5" />}
        </button>
        <input ref={fileInputRef} type="file" hidden accept=".pdf" onChange={handleFileChange} />
      </header>

      <div className="border-b border-rule p-4">
        <label className="flex h-9 items-center gap-2 border border-rule bg-paper px-3 focus-within:border-ink">
          <Search className="size-3.5 text-ink-faint" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search library"
            className="min-w-0 flex-1 bg-transparent text-xs outline-none placeholder:text-ink-faint"
          />
        </label>
      </div>

      <nav className="scroll-slim flex-1 overflow-x-hidden overflow-y-auto min-w-0 max-w-full px-3 py-4">
        <p className="label-meta px-2 pb-2">Workspace</p>
        <ul className="mb-6 space-y-0.5">
          <li>
            <button
              type="button"
              onClick={() => setTab("library")}
              className={cn(
                "flex w-full items-center gap-2.5 px-2 py-2 text-left text-xs",
                tab === "library" ? "bg-marker-soft font-medium text-marker" : "text-ink-soft hover:bg-canvas",
              )}
            >
              <Archive className="size-3.5" />
              <span className="flex-1">Library</span>
              <span className="font-mono text-[0.6rem] text-ink-faint">{pad(files.length)}</span>
            </button>
          </li>
          <li>
            <button
              type="button"
              onClick={() => setTab("recent")}
              className={cn(
                "flex w-full items-center gap-2.5 px-2 py-2 text-left text-xs",
                tab === "recent" ? "bg-marker-soft font-medium text-marker" : "text-ink-soft hover:bg-canvas",
              )}
            >
              <Clock3 className="size-3.5" />
              <span className="flex-1">Recent readings</span>
              <span className="font-mono text-[0.6rem] text-ink-faint">{pad(recents.length)}</span>
            </button>
          </li>
        </ul>

        <div className="flex items-center justify-between px-2 pb-2">
          <p className="label-meta">{listTitle}</p>
          {tab !== "library" && (
            <button
              type="button"
              onClick={() => setTab("library")}
              className="font-mono text-[0.6rem] text-ink-faint underline-offset-2 hover:text-marker hover:underline"
            >
              Show all
            </button>
          )}
        </div>

        {(uploadError || retryError || cancelError || reindexError) && (
          <div role="alert" className="mx-1 mb-2 border border-destructive/40 bg-destructive/5 px-3 py-2">
            <p className="text-xs font-medium text-destructive">
              {reindexError
                ? "Reindex failed"
                : cancelError
                  ? "Cancellation failed"
                  : retryError
                    ? "Retry failed"
                    : "Upload failed"}
            </p>
            <p className="mt-1 text-[0.65rem] leading-relaxed text-ink-soft">
              {reindexError ?? cancelError ?? retryError ?? uploadError}
            </p>
            <button
              type="button"
              onClick={() => {
                setUploadError(null);
                retryMutation.reset();
                cancelMutation.reset();
                reindexMutation.reset();
              }}
              className="mt-2 border border-rule bg-paper px-2 py-1 font-mono text-[0.6rem] hover:border-ink"
            >
              Dismiss
            </button>
          </div>
        )}

        {deleteMutation.isError && (
          <div role="alert" className="mx-1 mb-2 border border-destructive/40 bg-destructive/5 px-3 py-2">
            <p className="text-xs font-medium text-destructive">
              Delete failed{deleteTarget ? ` for ${displayTitle(deleteTarget)}` : ""}
            </p>
            <p className="mt-1 text-[0.65rem] leading-relaxed text-ink-soft">
              {deleteError ?? "The document was kept so you can retry."}
            </p>
            <div className="mt-2 flex gap-2">
              {deleteTarget && (
                <button
                  type="button"
                  onClick={() => deleteMutation.mutate(deleteTarget)}
                  disabled={deleteMutation.isPending}
                  className="border border-ink bg-ink px-2 py-1 font-mono text-[0.6rem] text-paper hover:bg-ink/90 disabled:opacity-40"
                >
                  Retry delete
                </button>
              )}
              <button
                type="button"
                onClick={() => deleteMutation.reset()}
                className="border border-rule bg-paper px-2 py-1 font-mono text-[0.6rem] hover:border-ink"
              >
                Dismiss
              </button>
            </div>
          </div>
        )}

        {filesQuery.isPending ? (
          <div className="space-y-2 px-1">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="h-[76px] w-full animate-pulse rounded-sm border border-rule bg-paper/60" />
            ))}
          </div>
        ) : filtered.length === 0 ? (
          <div className="mx-1 rounded-sm border border-dashed border-rule bg-paper/40 px-4 py-8 text-center">
            <div className="mx-auto grid size-8 place-items-center border border-rule bg-paper text-ink-faint">
              <File className="size-3.5" />
            </div>
            <p className="mt-3 text-xs font-medium">{emptyCopy.title}</p>
            <p className="mt-1 text-[0.65rem] leading-relaxed text-ink-faint">{emptyCopy.hint}</p>
            {tab === "library" && (
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="mt-4 inline-flex items-center gap-1.5 border border-ink bg-ink px-3 py-1.5 font-mono text-[0.65rem] text-paper hover:bg-ink/90"
              >
                <Upload className="size-3" /> Ingest Document
              </button>
            )}
            {tab !== "library" && (
              <button
                type="button"
                onClick={() => setTab("library")}
                className="mt-4 inline-flex items-center gap-1.5 border border-rule bg-paper px-3 py-1.5 font-mono text-[0.65rem] hover:border-ink"
              >
                Back to library
              </button>
            )}
          </div>
        ) : (
          <ul className="min-w-0 max-w-full space-y-px overflow-hidden">
            {filtered.map((item, index) => {
              const active = selectedFile?.id === item.id;
              const isRemoving = deleteMutation.isPending && (deleteMutation.variables as DbFile | undefined)?.id === item.id;
              const job = item.ingestion;
              const isRetrying = retryMutation.isPending && retryTarget === item.name;
              const isCancelling = cancelMutation.isPending && cancelTarget === item.name;
              const isJobActive = isIngestionActive(job?.state) || isRetrying || isCancelling;
              const isJobRetryable = isIngestionRetryable(job?.state) && !isRetrying;
              const isJobFailed = job?.state === "failed" && !isRetrying;
              const isReindexing = reindexMutation.isPending && reindexTarget === item.name;
              const isStaleIndex = isIndexStale(item.index) && !isJobActive && !isReindexing;

              const isDeleting = item.deletion_state === "deleting" || isRemoving;
              const isDeleteFailed = item.deletion_state === "delete_failed";

              return (
                <li key={item.id} className="min-w-0 max-w-full overflow-hidden">
                  <div
                      className={cn(
                        "group relative flex min-w-0 max-w-full items-center gap-0 overflow-hidden border-l-2 text-left transition-colors",
                        active ? "border-marker bg-paper" : "border-transparent hover:border-rule hover:bg-paper/70",
                        (isRemoving || isDeleting) && "opacity-50 pointer-events-none",
                      )}
                  >
                    <button
                      type="button"
                      onClick={() => {
                        // A failed job still has readable PDF bytes, so the
                        // reader stays reachable to inspect and retry it.
                        if (!isJobActive) selectFile(item);
                      }}
                      className={cn("flex min-w-0 max-w-full flex-1 flex-col gap-1 overflow-hidden px-3 py-3 text-left", isJobActive && "cursor-default")}
                    >
                      <span className="flex w-full min-w-0 max-w-full items-center justify-between overflow-hidden font-mono text-[0.58rem] text-ink-faint">
                        <span>0{index + 1}</span>
                        <span className="flex items-center gap-1.5">
                          {isDeleting ? (
                            <span className="inline-flex items-center gap-1">
                              <Loader2 className="size-2.5 animate-spin" />
                              Deleting
                            </span>
                          ) : isDeleteFailed ? (
                            <span className="text-destructive">Delete failed</span>
                          ) : isStaleIndex || isReindexing ? (
                            <span className="inline-flex items-center gap-1 text-destructive">
                              {isReindexing ? <Loader2 className="size-2.5 animate-spin" /> : <AlertTriangle className="size-2.5" />}
                              {isReindexing ? "Reindexing" : "Stale"}
                            </span>
                          ) : isJobFailed ? (
                            <span className="inline-flex items-center gap-1 text-destructive">
                              <AlertTriangle className="size-2.5" />
                              Failed
                            </span>
                          ) : isJobActive ? (
                            <span className="inline-flex items-center gap-1">
                              <Loader2 className="size-2.5 animate-spin" />
                              {isCancelling
                                ? "Cancelling"
                                : isRetrying
                                  ? "Retrying"
                                  : ingestionStageLabel(job?.stage ?? "queued")}
                            </span>
                          ) : isJobRetryable ? (
                            <span className="inline-flex items-center gap-1 text-destructive">
                              <AlertTriangle className="size-2.5" />
                              Retry
                            </span>
                          ) : (
                            <span>{formatFileSize(item.metadata.size)}</span>
                           )}
                         </span>
                       </span>

                      <span className={cn("block min-w-0 max-w-full overflow-hidden text-[0.78rem] leading-snug break-words line-clamp-2", active ? "font-medium text-ink" : "text-ink-soft group-hover:text-ink")}>
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
                      {isDeleteFailed ? (
                        <button
                          type="button"
                          onClick={() => deleteMutation.mutate(item)}
                          disabled={deleteMutation.isPending}
                          title={item.deletion_error ?? "Retry deletion"}
                          className="mr-1 border border-destructive/50 bg-paper px-2 py-1 font-mono text-[0.6rem] text-destructive hover:border-destructive disabled:opacity-40"
                        >
                          Retry
                        </button>
                      ) : isStaleIndex || isReindexing ? (
                        <button
                          type="button"
                          onClick={() => reindexMutation.mutate(item.name)}
                          disabled={isReindexing}
                          title={item.index?.change_details.map((c) => c.label).join(", ") ?? "Reindex"}
                          aria-label={`Reindex ${displayTitle(item)}`}
                          className="mr-1 inline-flex items-center gap-1 border border-destructive/50 bg-paper px-2 py-1 font-mono text-[0.6rem] text-destructive hover:border-destructive disabled:opacity-40"
                        >
                          {isReindexing ? <Loader2 className="size-3 animate-spin" /> : <RefreshCw className="size-3" />}
                          Reindex
                        </button>
                      ) : isJobRetryable ? (
                        <button
                          type="button"
                          onClick={() => retryMutation.mutate(item.name)}
                          disabled={isRetrying}
                          title={job?.error_message ?? "Retry indexing"}
                          aria-label={`Retry indexing ${displayTitle(item)}`}
                          className="mr-1 inline-flex items-center gap-1 border border-destructive/50 bg-paper px-2 py-1 font-mono text-[0.6rem] text-destructive hover:border-destructive disabled:opacity-40"
                        >
                          {isRetrying ? <Loader2 className="size-3 animate-spin" /> : <RotateCw className="size-3" />}
                          Retry
                        </button>
                      ) : isJobActive ? (
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
                              onClick={() => cancelMutation.mutate(item.name)}
                              disabled={isCancelling}
                              title="Cancel indexing"
                              aria-label={`Cancel indexing ${displayTitle(item)}`}
                              className="mr-1 grid size-7 place-items-center border border-rule bg-paper text-ink-soft hover:border-destructive hover:text-destructive disabled:opacity-40"
                            >
                              {isCancelling ? <Loader2 className="size-3 animate-spin" /> : <Square className="size-3" />}
                            </button>
                          )}
                        </>
                       ) : !isRemoving && !isDeleting ? (
                         <DropdownMenu>
                           <DropdownMenuTrigger asChild>
                             <button

                              type="button"
                              onClick={(e) => e.stopPropagation()}
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
                              onClick={() => deleteMutation.mutate(item)}
                            >
                              <Trash2 className="size-3.5" /> Remove Paper
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      ) : (
                        <span className="grid size-7 place-items-center text-ink-faint">
                          <Loader2 className="size-3.5 animate-spin" />
                        </span>
                      )}
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </nav>

      <footer className="flex justify-end border-t border-rule px-5 py-3">
        <button
          type="button"
          onClick={openSettings}
          aria-label="Open settings"
          className="grid size-7 place-items-center border border-rule bg-paper text-ink-faint hover:border-ink hover:text-ink"
          title="Settings"
        >
          <Settings className="size-3.5" />
        </button>
      </footer>
    </aside>
  );
}
