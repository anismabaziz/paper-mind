import { cn } from "@/lib/utils";

/**
 * The one line App Settings reports a save, a test, or a failed load with.
 *
 * Errors are announced rather than only shown, because the reader who pressed
 * Save is often looking at the button and not at the message above it.
 */
export default function FormFeedback({ kind, text }: { kind: "success" | "error"; text: string }) {
  return (
    <p
      data-testid="settings-feedback"
      role={kind === "error" ? "alert" : "status"}
      className={cn(
        "rounded-sm border px-3 py-2 font-mono text-xs",
        kind === "success"
          ? "border-ink/20 bg-canvas text-ink"
          : "border-destructive/20 bg-destructive/5 text-destructive"
      )}
    >
      {text}
    </p>
  );
}