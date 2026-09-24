import { useEffect, useMemo, useRef, useState } from "react";
import { Archive, Clock3, Search, Upload, File, Trash2, MoreHorizontal, Loader2, Settings } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { processFile } from "@/services/files";
import { useFiles, useUploadFile, useDeleteFile, useProcessFile, useTouchFileOpened } from "@/hooks/useFiles";
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
import type { File as DbFile } from "@/types/db";
import { displayTitle } from "@/types/db";

function pad(n: number) {
  return String(n).padStart(2, "0");
}

const MAX_RECENTS = 12;

export function LibraryRail() {
  const queryClient = useQueryClient();
  const { file: selectedFile, setFile } = usePdfStore();
  const [query, setQuery] = useState("");
  const [tab, setTab] = useState<"library" | "recent">("library");
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
    onSuccess: async (data) => {
      try {
        await processFile(data.file);
      } catch (err) {
        console.error("Indexing failed:", err);
        alert(`Indexing failed: ${err instanceof Error ? err.message : String(err)}`);
      } finally {
        // A fresh upload counts as opened so it enters Recent readings.
        touchOpened.mutate(data.file);
        // ["files"] prefix covers the per-document status/messages/meta keys.
        queryClient.invalidateQueries({ queryKey: ["files"] });
      }
    },
    onError: (err) => {
      alert(`Upload failed: ${err instanceof Error ? err.message : String(err)}`);
    },
  });

  const deleteMutation = useDeleteFile({
    onSuccess: (_data, variables) => {
      if (selectedFile?.id === variables.id) setFile(null);
    },
  });
  const deleteError = deleteMutation.error instanceof Error ? deleteMutation.error.message : null;
  const deleteTarget = deleteMutation.variables as DbFile | undefined;

  const processMutation = useProcessFile({
    onError: (err) => alert(`Indexing failed: ${err instanceof Error ? err.message : String(err)}`),
  });

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
              const isRetrying = processMutation.isPending && (processMutation.variables as DbFile | undefined)?.name === item.name;
              const isProcessing = !item.is_processed;
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
                        if (!isProcessing) selectFile(item);
                      }}
                      className={cn("flex min-w-0 max-w-full flex-1 flex-col gap-1 overflow-hidden px-3 py-3 text-left", isProcessing && "cursor-default")}
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
                          ) : isProcessing ? (
                            <span className="inline-flex items-center gap-1">
                              <Loader2 className="size-2.5 animate-spin" />
                              {isRetrying ? "Re-indexing" : "Indexing"}
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
                        {isDeleting ? "Deleting" : isDeleteFailed ? (item.deletion_error ?? "Delete failed — retry") : isProcessing ? "Queued for indexing" : "Indexed"}
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
                      ) : isProcessing ? (
                        <button
                          type="button"
                          onClick={() => processMutation.mutate(item)}
                          disabled={isRetrying}
                          className="mr-1 grid size-7 place-items-center border border-rule bg-paper text-[0.6rem] font-medium text-ink-soft hover:border-ink hover:text-ink disabled:opacity-40"
                        >
                          {isRetrying ? <Loader2 className="size-3 animate-spin" /> : "→"}
                        </button>
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
