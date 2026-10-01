import { useCallback, useEffect, useState } from "react";

// Single owner of a Document's fetched bytes. The source is never handed to
// pdf.js directly — each Renderer clones it (see lib/pdf-buffer) because the
// worker detaches whatever buffer it receives. Switching Documents or
// unmounting drops the source so previous buffers are never retained.
export function usePdfFileData(file: { id: string; url: string } | null) {
  const [data, setData] = useState<Uint8Array | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [generation, setGeneration] = useState(0);
  const fileId = file?.id ?? null;
  const fileUrl = file?.url ?? null;

  useEffect(() => {
    // Drop the previous Document's bytes synchronously so a switch never
    // retains two Documents at once, then fetch the new one.
    setData(null);
    setError(null);
    if (!fileUrl || !fileId) return;
    const url = fileUrl;
    let cancelled = false;
    const controller = new AbortController();
    async function load() {
      try {
        const res = await fetch(url, { signal: controller.signal });
        if (!res.ok) throw new Error(`Failed to load PDF (${res.status})`);
        const buf = await res.arrayBuffer();
        if (cancelled) return;
        // An empty body is a failed download, not an empty document: keep
        // the error visible with a retry instead of an endless loader.
        if (buf.byteLength === 0) throw new Error("The download came back empty.");
        setData(new Uint8Array(buf));
        setError(null);
      } catch (e) {
        if (cancelled || controller.signal.aborted) return;
        setData(null);
        setError(e instanceof Error ? e.message : "Failed to load PDF");
      }
    }
    load();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [fileId, fileUrl, generation]);

  // Refetch fresh bytes after pdf.js detached the source (e.g. the strip
  // remounted and its worker took the shared view). Stable across renders.
  // Reload also clears a download failure so a retry fetches again instead
  // of re-showing the same error.
  const reload = useCallback(() => {
    setError(null);
    setGeneration((g) => g + 1);
  }, []);

  // A pending load is `data === null && error === null`: one signal, so a
  // cancelled download can never leave a stale "loading" flag behind.
  const isLoading = fileUrl != null && data === null && error === null;

  return { data, error, isLoading, reload };
}
