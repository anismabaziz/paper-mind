import { Loader2, RefreshCw } from "lucide-react";
import { useReindex } from "@/hooks/useFiles";
import usePdfStore from "@/store/pdf-state";
import type { DocumentIndex } from "@/types/db";
import { formatManifestValue } from "./chat-message";

/**
 * Why this paper can no longer be searched, and what to do about it.
 *
 * The changes that caused it are listed rather than summarised: a reader whose
 * questions suddenly stop working needs to know which setting moved, because
 * that is the one they may want to put back.
 */
export default function StaleIndexNotice({ index }: { index: DocumentIndex }) {
  const file = usePdfStore((s) => s.file);
  const reindex = useReindex();

  return (
    <div
      role="status"
      className="rounded-sm border border-destructive/30 bg-destructive/5 p-5"
      data-testid="stale-index-notice"
    >
      <div className="flex items-center gap-2">
        <RefreshCw className="size-3.5 text-destructive" />
        <h4 className="font-mono text-[0.68rem] font-semibold uppercase tracking-widest text-destructive">
          Index needs reindexing
        </h4>
      </div>
      <p className="mt-2 font-serif text-xs leading-relaxed text-ink-soft">
        These settings changed after this paper was indexed, so its passages no longer match how
        the app searches. Reindex to ask questions again.
      </p>
      <dl className="mt-3 space-y-1.5">
        {index.change_details.map((change) => (
          <div key={change.field} className="flex flex-wrap items-baseline gap-x-2 text-[0.65rem]">
            <dt className="font-mono text-ink-faint">{change.label}</dt>
            <dd className="text-ink-soft">
              {change.indexed === null
                ? "not recorded"
                : `${formatManifestValue(change.indexed)} → ${formatManifestValue(change.current)}`}
            </dd>
          </div>
        ))}
      </dl>
      <p className="label-meta mt-3">
        Active index generation {index.manifest?.index_generation ?? "?"}
      </p>
      {reindex.isError && (
        <p className="mt-2 text-[0.65rem] text-destructive">
          {reindex.error instanceof Error ? reindex.error.message : "The reindex could not be queued."}
        </p>
      )}
      <button
        type="button"
        onClick={() => file && reindex.mutate(file.name)}
        disabled={reindex.isPending || !file}
        className="mt-3 inline-flex items-center gap-1.5 border border-ink bg-ink px-3 py-1.5 font-mono text-[0.65rem] text-paper hover:bg-ink/90 disabled:opacity-40"
        data-testid="chat-reindex"
      >
        {reindex.isPending ? <Loader2 className="size-3 animate-spin" /> : <RefreshCw className="size-3" />}
        {reindex.isPending ? "Queueing reindex…" : "Reindex this paper"}
      </button>
    </div>
  );
}