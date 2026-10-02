import { cn } from "@/lib/utils";

export type OutlineState = "loading" | "error" | "ready";

type Props = {
  outline: { title: string; page: number }[];
  state: OutlineState;
  errorText: string | null;
  activePage: number;
  onJump: (page: number) => void;
  onRetry: () => void;
  stripRef: React.RefObject<HTMLDivElement | null>;
  children: React.ReactNode;
};

/**
 * The Document's table of contents, above its page thumbnails.
 *
 * The entries come from the Document's own outline, which pdf.js reads out of
 * the file. A Document without one is not broken: it says so, rather than
 * showing an empty strip a reader would take for a failure to load.
 */
export default function OutlineStrip({
  outline,
  state,
  errorText,
  activePage,
  onJump,
  onRetry,
  stripRef,
  children,
}: Props) {
  return (
    <div className="shrink-0 border-b border-rule bg-background/50">
      <div className="border-b border-rule/60">
        <div
          ref={stripRef}
          className="flex items-center gap-2 overflow-x-auto px-4 py-2 [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden"
        >
          <span className="label-meta shrink-0 pr-2">Contents</span>
          {state === "error" ? (
            <span className="flex items-center gap-2">
              <span role="alert" className="font-mono text-[0.68rem] text-destructive">
                Outline unavailable — {errorText ?? "could not load outline."}
              </span>
              <button
                type="button"
                onClick={onRetry}
                className="border border-rule bg-paper px-2 py-0.5 font-mono text-[0.62rem] text-ink-soft hover:border-ink hover:text-ink"
              >
                Retry
              </button>
            </span>
          ) : outline.length === 0 ? (
            <span className="font-mono text-[0.68rem] text-ink-faint">
              {state === "loading" ? "Loading outline…" : "No outline"}
            </span>
          ) : (
            outline.map((entry, index) => (
              <button
                key={`${entry.title}-${entry.page}-${index}`}
                type="button"
                onClick={() => onJump(entry.page)}
                aria-label={`Jump to ${entry.title}, page ${entry.page}`}
                aria-current={activePage === entry.page ? "true" : undefined}
                className={cn(
                  "inline-flex shrink-0 items-center gap-1.5 rounded-full border px-3 py-1 text-xs leading-none transition-colors",
                  activePage === entry.page
                    ? "border-marker bg-marker-soft text-marker"
                    : "border-rule bg-paper text-ink-soft hover:border-ink hover:text-ink",
                )}
              >
                <span className="font-mono text-[0.62rem] text-ink-faint">{entry.page}</span>
                <span className="max-w-[18ch] truncate">{entry.title}</span>
              </button>
            ))
          )}
        </div>
      </div>
      {children}
    </div>
  );
}