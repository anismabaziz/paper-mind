import { Scale, Settings } from "lucide-react";

type Props = {
  briefCount: number;
  onOpenBrief: () => void;
  onOpenSettings: () => void;
};

/**
 * The two things the rail opens that are not Documents.
 *
 * The brief is disabled until two Documents could actually be part of one, and
 * says so while disabled. Offering it anyway would open the dialog straight
 * onto a refusal, which reads as the app refusing rather than as the pair not
 * being available yet.
 */
export default function RailFooter({ briefCount, onOpenBrief, onOpenSettings }: Props) {
  return (
    <footer className="flex items-center justify-between border-t border-rule px-5 py-3">
      <button
        type="button"
        onClick={onOpenBrief}
        disabled={briefCount < 2}
        aria-label="Start a research brief over two documents"
        title={briefCount < 2 ? "A brief compares two indexed documents" : "Compare two documents"}
        data-testid="open-research-brief"
        className="inline-flex items-center gap-1.5 font-mono text-[0.6rem] text-ink-faint hover:text-marker disabled:opacity-40"
      >
        <Scale size={12} /> Brief
      </button>
      <button
        type="button"
        onClick={onOpenSettings}
        aria-label="Open settings"
        className="grid size-7 place-items-center border border-rule bg-paper text-ink-faint hover:border-ink hover:text-ink"
        title="Settings"
      >
        <Settings className="size-3.5" />
      </button>
    </footer>
  );
}