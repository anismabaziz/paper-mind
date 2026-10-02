import { CornerDownLeft, FileText, Loader2 } from "lucide-react";
import { FailureNotice } from "@/components/FailureNotice";
import { suggestedPrompts } from "./chat-message";

type Props = {
  hasFile: boolean;
  /** Set when the status request failed with nothing to fall back on. */
  statusError: { text: string | null; onRetry: () => void } | null;
  /** Set when the status request failed but an earlier answer is still shown. */
  statusStale: boolean;
  onRetryStatus: () => void;
  /** The paper is searchable. */
  ready: boolean;
  indexing: boolean;
  showStaleIndex: boolean;
  conversationError: { text: string | null; onRetry: () => void } | null;
  showPrompts: boolean;
  onPrompt: (prompt: string) => void;
};

/**
 * What the conversation shows when there is no conversation to show.
 *
 * These are different situations that all render as an empty pane: no paper
 * chosen, no confirmed state of the paper, the paper indexing, the index no
 * longer matching, past turns unavailable, or genuinely nothing asked yet. Each
 * says which one it is, because a spinner shown for all of them reads as a hang.
 */
export default function ChatConversationStates({
  hasFile,
  statusError,
  statusStale,
  onRetryStatus,
  ready,
  indexing,
  showStaleIndex,
  conversationError,
  showPrompts,
  onPrompt,
}: Props) {
  if (!hasFile) {
    return (
      <div className="flex flex-col items-center py-16 text-center">
        <div className="grid size-10 place-items-center border border-rule bg-paper text-ink-faint">
          <FileText className="size-4" />
        </div>
        <h4 className="mt-4 font-serif text-sm font-medium">No Active Session</h4>
        <p className="mt-1 max-w-[22ch] font-serif text-xs leading-relaxed text-ink-faint">
          Select a research paper from your library to start an interactive analysis session.
        </p>
      </div>
    );
  }

  return (
    <>
      {statusError && (
        <FailureNotice
          testId="chat-status-error"
          title="Could not check this paper"
          message={`The last confirmed state is not available, so questions are paused instead of guessing. ${statusError.text ?? "The status request failed."}`}
          actionLabel="Retry status check"
          onAction={statusError.onRetry}
        />
      )}

      {statusStale && (
        <FailureNotice
          testId="chat-status-stale"
          variant="stale"
          title="Showing the last confirmed state"
          message="This paper's status could not be refreshed. Nothing changed on the server, and questions stay available."
          actionLabel="Refresh status"
          onAction={onRetryStatus}
        />
      )}

      {!statusError && !statusStale && showStaleIndex === false && (!ready || indexing) && (
        <div className="flex flex-col items-center py-16 text-center">
          <Loader2 className="size-6 animate-spin text-ink-faint" />
          <h4 className="mt-4 font-serif text-sm font-medium">
            {indexing ? "Reindexing Document…" : "Indexing Document…"}
          </h4>
          <p className="mt-1 max-w-[26ch] text-xs leading-relaxed text-ink-faint">
            Generating semantic vector representations for retrieval-augmented analysis.
          </p>
        </div>
      )}

      {conversationError && (
        <FailureNotice
          testId="chat-conversation-error"
          title="Could not load this Conversation"
          message={`Past turns are unavailable right now — new questions still work. ${conversationError.text ?? "The request failed."}`}
          actionLabel="Retry loading turns"
          onAction={conversationError.onRetry}
        />
      )}

      {showPrompts && (
        <div className="rounded-sm border border-rule bg-paper p-5 text-center shadow-sm">
          <h4 className="font-mono text-[0.68rem] font-semibold uppercase tracking-widest">
            Session Initialized
          </h4>
          <p className="mx-auto mt-2 max-w-[30ch] font-serif text-xs leading-relaxed text-ink-faint">
            Select a query template below or enter a custom prompt in the input workbench.
          </p>
          <div className="mt-4 space-y-1.5 text-left">
            {suggestedPrompts.map((prompt) => (
              <button
                key={prompt}
                type="button"
                onClick={() => onPrompt(prompt)}
                className="flex w-full items-center justify-between rounded-sm border border-rule bg-canvas px-3 py-2 text-xs text-ink-soft transition-colors hover:border-ink hover:text-ink"
              >
                <span className="truncate">{prompt}</span>
                <CornerDownLeft className="size-3 opacity-50" />
              </button>
            ))}
          </div>
        </div>
      )}
    </>
  );
}