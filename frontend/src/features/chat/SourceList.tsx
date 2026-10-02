import { useState } from "react";
import { ChevronDown } from "lucide-react";
import usePdfStore from "@/store/pdf-state";
import { cn } from "@/lib/utils";
import type { IRetrievalResult, ISource } from "@/services/files";
import { METHOD_LABELS } from "./chat-message";

/**
 * The passages an answer was read from.
 *
 * Every one of them is a jump into the reader, because an answer the reader
 * cannot check is an answer they have to take on trust. A passage with no page
 * still shows, marked as not jumpable, so the citation count stays honest.
 */
export default function SourceList({
  sources,
  retrieval,
}: {
  sources: ISource[];
  retrieval?: IRetrievalResult;
}) {
  const [open, setOpen] = useState(true);
  const setCitationTarget = usePdfStore((s) => s.setCitationTarget);
  const listId = `source-list-${sources[0]?.source_id ?? "empty"}`;

  return (
    <div className="mt-4 border-t border-rule pt-3">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-controls={listId}
        aria-label={`${open ? "Hide" : "Show"} ${sources.length} cited passages`}
        className="flex w-full items-center justify-between text-ink-faint hover:text-ink"
      >
        <span className="label-meta">
          {sources.length} passages{retrieval ? ` · ${METHOD_LABELS[retrieval.method]}` : ""}
        </span>
        <ChevronDown className={cn("size-3 transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <ol id={listId} className="mt-3 space-y-3">
          {sources.map((source, idx) => {
            const hasPage = source.page != null;
            return (
              <li key={idx}>
                <button
                  type="button"
                  disabled={!hasPage}
                  data-testid={`source-jump-${source.source_id}`}
                  onClick={() => {
                    if (hasPage) setCitationTarget(source.page);
                  }}
                  className={cn(
                    "group block w-full border-l-2 py-0.5 pl-3 text-left transition-colors",
                    hasPage
                      ? "border-rule hover:border-marker cursor-pointer"
                      : "border-rule cursor-default",
                  )}
                  title={hasPage ? `Jump to page ${source.page}` : undefined}
                >
                  <span className="flex items-baseline justify-between gap-2">
                    <span className="font-mono text-[0.62rem] tracking-wide text-ink-faint">
                      <span className="text-marker">[{source.source_id}]</span> {source.document} ·
                      chunk {source.chunk_index}
                      {hasPage ? ` · p. ${source.page}` : ""}
                    </span>
                    <span className="font-mono text-[0.6rem] text-ink-faint">
                      Rank {source.rank ?? idx + 1}
                    </span>
                  </span>
                  <span className="mt-1 block font-serif text-[0.85rem] leading-snug text-ink-soft italic">
                    “{source.content.slice(0, 220)}”
                  </span>
                </button>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}