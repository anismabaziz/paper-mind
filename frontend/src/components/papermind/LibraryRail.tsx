import { useEffect, useMemo, useRef, useState } from "react";
import { Archive, Clock3, Folder, Plus, Search, Upload, File, Trash2, MoreHorizontal, Loader2, Settings } from "lucide-react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getFiles, uploadFile, deleteFile, processFile } from "@/services/files";
import usePdfStore from "@/store/pdf-state";
import useSettingsUi from "@/store/settings-ui";
import { cn } from "@/lib/utils";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu";
import type { File as DbFile } from "@/types/db";
import { displayTitle } from "@/types/db";

export function LibraryRail() {
  const queryClient = useQueryClient();
  const { file: selectedFile, setFile } = usePdfStore();
  const [query, setQuery] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const openSettings = useSettingsUi((s) => s.open);
  const filesQuery = useQuery({
    queryKey: ["files"],
    queryFn: getFiles,
    refetchInterval: (q) => {
      const hasUnprocessed = q.state.data?.files.some((f) => !f.is_processed);
      return hasUnprocessed ? 3000 : false;
    },
  });

  const files = filesQuery.data?.files ?? [];

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
  }, [files, selectedFile, setFile]);

  const filtered = useMemo(() => {
    const v = query.trim().toLowerCase();
    if (!v) return files;
    return files.filter((f) => displayTitle(f).toLowerCase().includes(v));
  }, [files, query]);

  const uploadMutation = useMutation({
    mutationFn: uploadFile,
    onSuccess: async (data) => {
      queryClient.invalidateQueries({ queryKey: ["files"] });
      try {
        await processFile(data.file);
      } catch (err) {
        console.error("Indexing failed:", err);
        alert(`Indexing failed: ${err instanceof Error ? err.message : String(err)}`);
      } finally {
        queryClient.invalidateQueries({ queryKey: ["files"] });
        queryClient.invalidateQueries({ queryKey: [data.file.name, "is-processed"] });
      }
    },
    onError: (err) => {
      alert(`Upload failed: ${err instanceof Error ? err.message : String(err)}`);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: deleteFile,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["files"] }),
  });

  const processMutation = useMutation({
    mutationFn: processFile,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["files"] }),
    onError: (err) => alert(`Indexing failed: ${err instanceof Error ? err.message : String(err)}`),
  });

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files?.[0]) uploadMutation.mutate(e.target.files[0]);
    if (e.target) e.target.value = "";
  }

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1048576) return (bytes / 1024).toFixed(0) + " KB";
    return (bytes / 1048576).toFixed(1) + " MB";
  };

  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-rule bg-sidebar">
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

      <nav className="scroll-slim flex-1 overflow-y-auto px-3 py-4">
        <p className="label-meta px-2 pb-2">Workspace</p>
        <ul className="mb-6 space-y-0.5">
          {(
            [
              [Archive, "Library", String(files.length).padStart(2, "0")],
              [Clock3, "Recent readings", "12"],
              [Folder, "Collections", "03"],
            ] as const
          ).map(([Icon, label, count], index) => (
            <li key={label}>
              <button
                type="button"
                className={cn(
                  "flex w-full items-center gap-2.5 px-2 py-2 text-left text-xs",
                  index === 0 ? "bg-marker-soft font-medium text-marker" : "text-ink-soft hover:bg-canvas",
                )}
              >
                <Icon className="size-3.5" />
                <span className="flex-1">{String(label)}</span>
                <span className="font-mono text-[0.6rem] text-ink-faint">{String(count)}</span>
              </button>
            </li>
          ))}
        </ul>

        <div className="flex items-center justify-between px-2 pb-2">
          <p className="label-meta">Current project</p>
          <button type="button" className="text-ink-faint hover:text-marker" aria-label="New project" title="New project">
            <Plus className="size-3.5" />
          </button>
        </div>
        <p className="mb-4 px-2 text-xs font-medium">Agent governance / 2026</p>

        <p className="label-meta px-2 pb-2">Papers · {filtered.length}</p>

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
            <p className="mt-3 text-xs font-medium">No documents</p>
            <p className="mt-1 text-[0.65rem] leading-relaxed text-ink-faint">Ingest a PDF to begin analysis.</p>
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="mt-4 inline-flex items-center gap-1.5 border border-ink bg-ink px-3 py-1.5 font-mono text-[0.65rem] text-paper hover:bg-ink/90"
            >
              <Upload className="size-3" /> Ingest Document
            </button>
          </div>
        ) : (
          <ul className="space-y-px">
            {filtered.map((item, index) => {
              const active = selectedFile?.id === item.id;
              const isRemoving = deleteMutation.isPending && (deleteMutation.variables as DbFile | undefined)?.id === item.id;
              const isRetrying = processMutation.isPending && (processMutation.variables as DbFile | undefined)?.name === item.name;
              const isProcessing = !item.is_processed;

              return (
                <li key={item.id}>
                  <div
                    className={cn(
                      "group relative flex items-center gap-0 border-l-2 text-left transition-colors",
                      active ? "border-marker bg-paper" : "border-transparent hover:border-rule hover:bg-paper/70",
                      isRemoving && "opacity-50 pointer-events-none",
                    )}
                  >
                    <button
                      type="button"
                      onClick={() => !isProcessing && setFile(item)}
                      className={cn("flex flex-1 flex-col gap-1 px-3 py-3 text-left", isProcessing && "cursor-default")}
                    >
                      <span className="flex w-full items-center justify-between font-mono text-[0.58rem] text-ink-faint">
                        <span>0{index + 1}</span>
                        <span className="flex items-center gap-1.5">
                          {isProcessing ? (
                            <span className="inline-flex items-center gap-1">
                              <Loader2 className="size-2.5 animate-spin" />
                              {isRetrying ? "Re-indexing" : "Indexing"}
                            </span>
                          ) : (
                            <span>{formatSize(item.metadata.size)}</span>
                          )}
                        </span>
                      </span>
                      <span className={cn("block text-[0.78rem] leading-snug line-clamp-2", active ? "font-medium text-ink" : "text-ink-soft group-hover:text-ink")}>
                        {displayTitle(item)}
                      </span>
                      <span className="block truncate text-[0.65rem] text-ink-faint">
                        {item.metadata.content_type.split("/").pop()?.toUpperCase() ?? "PDF"} ·{" "}
                        {isProcessing ? "Queued for indexing" : "Indexed"}
                      </span>
                    </button>

                    <div className="pr-1">
                      {isProcessing ? (
                        <button
                          type="button"
                          onClick={() => processMutation.mutate(item)}
                          disabled={isRetrying}
                          className="mr-1 grid size-7 place-items-center border border-rule bg-paper text-[0.6rem] font-medium text-ink-soft hover:border-ink hover:text-ink disabled:opacity-40"
                        >
                          {isRetrying ? <Loader2 className="size-3 animate-spin" /> : "→"}
                        </button>
                      ) : !isRemoving ? (
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

      <footer className="border-t border-rule px-5 py-4">
        <div className="flex items-center gap-3">
          <span className="grid size-8 place-items-center rounded-full bg-canvas font-mono text-[0.6rem] font-bold">AT</span>
          <div className="min-w-0 flex-1">
            <p className="truncate text-xs font-medium">Aris Thorne</p>
            <p className="text-[0.62rem] text-ink-faint">Portfolio prototype</p>
          </div>
          <button
            type="button"
            onClick={openSettings}
            aria-label="Open settings"
            className="grid size-7 place-items-center border border-rule bg-paper text-ink-faint hover:border-ink hover:text-ink"
            title="Settings"
          >
            <Settings className="size-3.5" />
          </button>
        </div>
      </footer>
    </aside>
  );
}
