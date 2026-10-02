import { useEffect, useMemo, useState } from "react";
import { displayTitle } from "@/types/db";
import type { File as DbFile } from "@/types/db";

const MAX_RECENTS = 12;

type Args = {
  files: DbFile[];
  selectedFile: DbFile | null;
  /** A reader opened this Document, so it counts towards recent readings. */
  onOpen: (file: DbFile) => void;
  /** The app opened this one, which recent readings must not record. */
  onFallback: (file: DbFile) => void;
  onDeselect: () => void;
};

/**
 * What the rail is showing: the whole library, the recent readings, and which
 * of them the reader's search and tab leave visible.
 *
 * One Document stays selected on purpose. Switching to Recent readings does not
 * close the reader, because a reader who narrows the list to find a paper they
 * read yesterday wants to land on it, not on a blank pane.
 */
export function useLibraryListing({ files, selectedFile, onOpen, onFallback, onDeselect }: Args) {
  const [query, setQuery] = useState("");
  const [tab, setTab] = useState<"library" | "recent">("library");

  const recents = useMemo(() => {
    const opened = files.filter((f): f is DbFile & { last_opened_at: string } => f.last_opened_at != null);
    return opened
      .sort((a, b) => b.last_opened_at.localeCompare(a.last_opened_at))
      .slice(0, MAX_RECENTS);
  }, [files]);

  useEffect(() => {
    if (!files.length) {
      if (selectedFile) onDeselect();
      return;
    }
    const exists = selectedFile && files.find((f) => f.id === selectedFile.id);
    // The first Document a reader could actually read, opened by the app
    // rather than by them, so it stays out of the recent-readings order.
    if (!exists) onFallback(files.find((f) => f.is_processed) ?? files[0]);
  }, [files, onDeselect, onFallback, selectedFile]);

  const viewFiles: DbFile[] = tab === "recent" ? recents : files;
  const filtered = useMemo(() => {
    const wanted = query.trim().toLowerCase();
    if (!wanted) return viewFiles;
    return viewFiles.filter((f) => displayTitle(f).toLowerCase().includes(wanted));
  }, [query, viewFiles]);

  return {
    onOpen,
    tab,
    setTab,
    query,
    setQuery,
    recents,
    filtered,
    listTitle: tab === "recent" ? `Recent · ${filtered.length}` : `Papers · ${filtered.length}`,
  };
}