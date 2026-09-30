import { useCallback, useEffect, useState } from "react";

// Single owner of a Document's fetched bytes. The source is never handed to
// pdf.js directly — each Renderer clones it (see lib/pdf-buffer) because the
// worker detaches whatever buffer it receives. Switching Documents or
// unmounting drops the source so previous buffers are never retained.
export function usePdfFileData(file: { id: string; url: string } | null) {
  const [data, setData] = useState<Uint8Array | null>(null);
  const [generation, setGeneration] = useState(0);
  const fileId = file?.id ?? null;
  const fileUrl = file?.url ?? null;

  useEffect(() => {
    // Drop the previous Document's bytes synchronously so a switch never
    // retains two Documents at once, then fetch the new one.
    setData(null);
    if (!fileUrl || !fileId) return;
    const url = fileUrl;
    let cancelled = false;
    const controller = new AbortController();
    async function load() {
      try {
        const res = await fetch(url, { signal: controller.signal });
        if (!res.ok) throw new Error(`Failed to load PDF (${res.status})`);
        const buf = await res.arrayBuffer();
        if (!cancelled) setData(new Uint8Array(buf));
      } catch {
        if (!cancelled) setData(null);
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
  const reload = useCallback(() => setGeneration((g) => g + 1), []);

  return { data, reload };
}
