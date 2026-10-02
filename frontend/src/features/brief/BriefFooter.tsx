import { Loader2, Search, Square } from "lucide-react";

/**
 * What the brief dialog offers, which is one button at any moment.
 *
 * While a brief runs, the start button becomes the only control that says
 * anything: it reads "Reading…" and is disabled, so the reader cannot start a
 * second brief over the same pair.
 */
export default function BriefFooter({
  running,
  canStart,
  onStart,
  onStop,
}: {
  running: boolean;
  canStart: boolean;
  onStart: () => void;
  onStop: () => void;
}) {
  return (
    <footer className="flex items-center justify-between gap-2 border-t border-rule px-5 py-3">
      {running ? (
        <button
          type="button"
          onClick={onStop}
          data-testid="brief-stop"
          className="inline-flex items-center gap-1.5 border border-rule bg-paper px-3 py-1.5 font-mono text-[0.65rem] text-ink-soft hover:border-destructive hover:text-destructive"
        >
          <Square size={12} /> Stop
        </button>
      ) : (
        <p className="label-meta">Reads only the two Documents above.</p>
      )}
      <button
        type="button"
        onClick={onStart}
        disabled={!canStart || running}
        data-testid="brief-start"
        className="inline-flex h-11 min-h-[44px] items-center gap-1.5 bg-ink px-4 font-mono text-[0.65rem] text-paper hover:bg-ink/90 disabled:opacity-40"
      >
        {running ? <Loader2 size={12} className="animate-spin" /> : <Search size={12} />}
        {running ? "Reading…" : "Start brief"}
      </button>
    </footer>
  );
}
