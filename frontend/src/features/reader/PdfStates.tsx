import { RotateCw } from "lucide-react";
import { FailureNotice } from "@/components/FailureNotice";

/**
 * What the sheet shows while pdf.js is working, and what it shows when it
 * cannot.
 *
 * A failed render is recoverable, so it says what is still true — the Document
 * is in the library, and rendering it again usually clears this — rather than
 * presenting a dead canvas.
 */
export function PdfLoading({ label = "Loading document…" }: { label?: string }) {
  return (
    <div className="grid h-[760px] place-items-center bg-white">
      <p className="font-mono text-xs text-ink-faint">{label}</p>
    </div>
  );
}

export function PdfError({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div
      className="grid h-[760px] place-items-center bg-white p-6 text-center"
      data-testid="reader-render-error"
    >
      <div>
        <FailureNotice
          title="Could not render this document"
          message={`${message} The document is still in your library — rendering it again usually clears this.`}
        />
        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="mt-3 inline-flex items-center gap-1.5 border border-ink bg-ink px-3 py-1.5 font-mono text-[0.65rem] text-paper hover:bg-ink/90"
          >
            <RotateCw className="size-3" /> Retry rendering
          </button>
        )}
      </div>
    </div>
  );
}