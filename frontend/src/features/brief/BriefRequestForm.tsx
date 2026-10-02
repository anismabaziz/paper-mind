import { FailureNotice } from "@/components/FailureNotice";
import { Label } from "@/components/ui/label";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import type { File as DbFile } from "@/types/db";
import type { IBriefDocument } from "@/services/research";
import BriefScopePicker from "./BriefScopePicker";

type Props = {
  /** What the library request is doing, which decides what this shows. */
  library: "loading" | "failed" | "tooFew" | "ready";
  libraryError: string | null;
  onRetryLibrary: () => void;
  files: DbFile[];
  selected: string[];
  onToggle: (file: DbFile) => void;
  scope: IBriefDocument[] | null;
  failure: string | null;
  question: string;
  onQuestion: (value: string) => void;
  disabled: boolean;
  /** Why a brief cannot start yet, once there is one to show. */
  reason: string | null;
};

/**
 * What a brief needs before it can start: two Documents, a question, and a
 * reason if either is not yet there.
 *
 * The reason is only shown once App Settings have loaded. Read off a
 * half-loaded catalog it would name the model when the reader's problem is
 * their Documents, and the other way round.
 */
export default function BriefRequestForm({
  library,
  libraryError,
  onRetryLibrary,
  files,
  selected,
  onToggle,
  scope,
  failure,
  question,
  onQuestion,
  disabled,
  reason,
}: Props) {
  return (
    <>
      {library === "loading" && (
        <p role="status" className="py-8 text-center font-mono text-xs text-ink-faint">
          <Spinner className="mr-2 inline size-3.5" aria-hidden="true" />
          Loading your library…
        </p>
      )}

      {library === "failed" && (
        <FailureNotice
          testId="brief-library-error"
          title="Could not load your library"
          message={`The Documents a brief could read are unavailable, so none were selected. ${libraryError ?? "The library request failed."}`}
          actionLabel="Retry library load"
          onAction={onRetryLibrary}
        />
      )}

      {library === "tooFew" && (
        <p
          className="py-8 text-center font-serif text-xs leading-relaxed text-ink-faint"
          data-testid="brief-not-enough"
        >
          A brief compares two indexed Documents. Index at least two before starting one.
        </p>
      )}

      {library === "ready" && (
        <BriefScopePicker files={files} selected={selected} onToggle={onToggle} scope={scope} />
      )}

      {failure && (
        <FailureNotice testId="brief-error" title="This brief could not start" message={failure} />
      )}

      <div className="mt-4 space-y-1">
        <Label htmlFor="brief-question" className="label-meta">
          Cross-document question
        </Label>
        <Textarea
          id="brief-question"
          rows={3}
          value={question}
          onChange={(event) => onQuestion(event.target.value)}
          disabled={disabled}
          aria-label="Cross-document question"
          placeholder="Where do these two papers disagree?"
          data-testid="brief-question"
          className="resize-y rounded-sm border-rule bg-card px-3 py-2 text-xs leading-relaxed text-ink focus-visible:border-ink focus-visible:ring-0 disabled:opacity-60"
        />
      </div>

      {reason && (
        <p className="label-meta mt-2" data-testid="brief-reason">
          {reason}
        </p>
      )}
    </>
  );
}