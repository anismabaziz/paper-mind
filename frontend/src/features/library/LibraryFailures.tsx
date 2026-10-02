import type { File as DbFile } from "@/types/db";
import { displayTitle } from "@/types/db";

export type LibraryFailure = {
  key: "upload" | "retry" | "cancel" | "reindex";
  heading: string;
  detail: string;
  reset: () => void;
};

/**
 * Each failure keeps its own heading and its own dismissal.
 *
 * Collapsing them into one priority chain hid an upload failure behind an
 * unrelated reindex error, so neither read as what it was: a reader was told
 * the wrong thing failed, with the wrong action attached.
 */
export default function LibraryFailures({ failures }: { failures: LibraryFailure[] }) {
  if (failures.length === 0) return null;
  return (
    <>
      {failures.map((failure) => (
        <div
          key={failure.key}
          role="alert"
          className="mx-1 mb-2 border border-destructive/40 bg-destructive/5 px-3 py-2"
          data-testid={`library-${failure.key}-error`}
        >
          <p className="text-xs font-medium text-destructive">{failure.heading}</p>
          <p className="mt-1 text-[0.65rem] leading-relaxed text-ink-soft">{failure.detail}</p>
          <button
            type="button"
            onClick={failure.reset}
            className="mt-2 border border-rule bg-paper px-2 py-1 font-mono text-[0.6rem] hover:border-ink"
          >
            Dismiss
          </button>
        </div>
      ))}
    </>
  );
}

/**
 * A removal that failed is offered back rather than reported as done: the
 * Document is still in the library, and retrying is the action that clears it.
 */
export function DeleteFailure({
  target,
  detail,
  pending,
  onRetry,
  onDismiss,
}: {
  target: DbFile | undefined;
  detail: string | null;
  pending: boolean;
  onRetry: () => void;
  onDismiss: () => void;
}) {
  return (
    <div role="alert" className="mx-1 mb-2 border border-destructive/40 bg-destructive/5 px-3 py-2">
      <p className="text-xs font-medium text-destructive">
        Delete failed{target ? ` for ${displayTitle(target)}` : ""}
      </p>
      <p className="mt-1 text-[0.65rem] leading-relaxed text-ink-soft">
        {detail ?? "The document was kept so you can retry."}
      </p>
      <div className="mt-2 flex gap-2">
        {target && (
          <button
            type="button"
            onClick={onRetry}
            disabled={pending}
            className="border border-ink bg-ink px-2 py-1 font-mono text-[0.6rem] text-paper hover:bg-ink/90 disabled:opacity-40"
          >
            Retry delete
          </button>
        )}
        <button
          type="button"
          onClick={onDismiss}
          className="border border-rule bg-paper px-2 py-1 font-mono text-[0.6rem] hover:border-ink"
        >
          Dismiss
        </button>
      </div>
    </div>
  );
}