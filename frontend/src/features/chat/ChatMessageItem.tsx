import { Loader2, RefreshCw, RotateCw, Settings } from "lucide-react";
import { MarkdownRenderer } from "@/components/MarkdownRenderer";
import useSettingsUi from "@/store/settings-ui";
import { cn } from "@/lib/utils";
import ClaimList from "./ClaimList";
import SourceList from "./SourceList";
import { ABSTENTION_HINTS, describe, FAILURE_COPY, type ChatMessage } from "./chat-message";

type Props = {
  message: ChatMessage;
  /** False while the paper is indexing, which no recovery can beat. */
  canRetry: boolean;
  busy: boolean;
  onRetry: (question: string) => void;
  reindexPending: boolean;
  onReindex: () => void;
};

/**
 * One answer, from the label above it to the passages under it.
 *
 * Every way an answer can end is rendered as itself: still arriving, stopped,
 * truncated, abstained, or failed. A failed answer never shows citations, and a
 * stopped one is never presented as stored, because the reader would otherwise
 * take the transcript for what will still be in the Conversation later.
 */
export default function ChatMessageItem({
  message,
  canRetry,
  busy,
  onRetry,
  reindexPending,
  onReindex,
}: Props) {
  const outcome = describe(message);
  // Narrowed once so the retry button never passes an undefined question to a
  // new turn.
  const retryQuestion = message.question;
  const copy = message.failure ? FAILURE_COPY[message.failure] : null;

  return (
    <div className="rise-in" data-outcome={outcome.outcome} data-testid="chat-message" data-sender="bot">
      <p className="label-meta mb-2">Synthesis · {outcome.label}</p>
      {message.cancelled && !message.text && (
        <p className="font-serif text-xs italic text-ink-faint">
          {message.stopReason
            ? `This answer was stopped (${message.stopReason}) before it finished, so it was not saved.`
            : "This answer was stopped before it finished, so it was not saved."}
        </p>
      )}
      {message.abstained && message.abstentionReason && (
        <p className="mb-1.5 font-mono text-[0.62rem] leading-relaxed tracking-wide text-ink-faint">
          {ABSTENTION_HINTS[message.abstentionReason]}
        </p>
      )}
      {message.abstained && message.abstentionReason === "evidence_unusable" && (
        <button
          type="button"
          onClick={onReindex}
          disabled={reindexPending}
          className="mb-1.5 inline-flex items-center gap-1.5 border border-rule px-2.5 py-1 font-mono text-[0.62rem] uppercase tracking-wide text-ink-soft transition-colors hover:border-ink hover:text-ink disabled:opacity-40"
        >
          {reindexPending ? (
            <Loader2 className="size-3 animate-spin" />
          ) : (
            <RefreshCw className="size-3" />
          )}
          {reindexPending ? "Queueing reindex…" : "Reindex this paper"}
        </button>
      )}
      {message.truncated && !message.failed && !message.cancelled && (
        <p className="mb-1.5 font-mono text-[0.62rem] tracking-wide text-ink-faint">
          The model hit its answer limit, so this stops mid-thought.
        </p>
      )}
      {message.failed && copy && (
        <>
          <p className="font-mono text-[0.62rem] font-semibold tracking-wide text-destructive">
            {copy.heading}
          </p>
          <p className="mt-1 font-mono text-[0.62rem] leading-relaxed tracking-wide text-ink-soft">
            {copy.hint}
          </p>
        </>
      )}
      <div
        role={message.failed ? "alert" : undefined}
        className={cn(
          "rounded-sm border px-3.5 py-3",
          message.failed
            ? "border-destructive/30 bg-destructive/5 text-destructive"
            : "border-transparent bg-transparent px-0 py-0",
        )}
      >
        {message.text ? (
          <>
            <MarkdownRenderer text={message.text} />
            {message.needsSettings && <OpenSettingsLink />}
            {message.failed && copy && retryQuestion && (
              <button
                type="button"
                onClick={() => onRetry(retryQuestion)}
                disabled={busy || !canRetry}
                className="mt-2 inline-flex items-center gap-1.5 border border-destructive/50 bg-paper px-2.5 py-1 font-mono text-[0.62rem] text-destructive hover:border-destructive disabled:opacity-40"
              >
                <RotateCw className="size-3" /> {copy.retryLabel}
              </button>
            )}
          </>
        ) : (
          <span className="flex items-center gap-1.5 py-1">
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink/20" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink/20 [animation-delay:0.2s]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-ink/20 [animation-delay:0.4s]" />
          </span>
        )}
      </div>
      {!message.failed && message.sources && message.sources.length > 0 && (
        <SourceList sources={message.sources} retrieval={message.retrieval} />
      )}
      {!message.failed && message.claims && message.claims.length > 0 && message.sources && (
        <ClaimList claims={message.claims} sources={message.sources} />
      )}
    </div>
  );
}

function OpenSettingsLink() {
  const openSettings = useSettingsUi((s) => s.open);
  return (
    <button
      onClick={openSettings}
      className="mt-2 inline-flex items-center gap-1 font-mono text-[0.62rem] font-semibold tracking-wide text-destructive underline underline-offset-2 hover:text-destructive/80"
    >
      <Settings className="size-3" /> Open Settings
    </button>
  );
}