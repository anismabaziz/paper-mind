import { RotateCw } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * The one way this app reports a failed request inside a workflow.
 *
 * A failure is a heading the server did not write, an explanation safe to
 * show, and — when the reader can do something — a single recovery action.
 * It never replaces the workflow around it: the banner sits above the
 * document, library, or Conversation that failed, so context survives.
 */
export function FailureNotice({
  title,
  message,
  actionLabel,
  onAction,
  variant = "error",
  testId,
  actionTestId,
}: {
  title: string;
  message: string;
  actionLabel?: string;
  onAction?: () => void;
  /** `error` interrupts; `stale` says a known state is being shown anyway. */
  variant?: "error" | "stale";
  testId?: string;
  actionTestId?: string;
}) {
  return (
    <div
      role={variant === "error" ? "alert" : "status"}
      data-testid={testId}
      className={cn(
        "rounded-sm border px-5 py-3",
        variant === "error"
          ? "border-destructive/40 bg-destructive/5"
          : "border-rule bg-paper",
      )}
    >
      <p
        className={cn(
          "font-mono text-[0.68rem] font-semibold uppercase tracking-widest",
          variant === "error" ? "text-destructive" : "text-ink-soft",
        )}
      >
        {title}
      </p>
      <p className="mt-1.5 text-[0.68rem] leading-relaxed text-ink-soft">{message}</p>
      {actionLabel && onAction && (
        <button
          type="button"
          onClick={onAction}
          data-testid={actionTestId}
          className="mt-2.5 inline-flex items-center gap-1.5 border border-ink bg-ink px-3 py-1.5 font-mono text-[0.65rem] text-paper hover:bg-ink/90"
        >
          <RotateCw className="size-3" /> {actionLabel}
        </button>
      )}
    </div>
  );
}
