import {
  chatStream,
  classifyChatRequestError,
  isSettingsFailureMessage,
} from "@/services/files";
import type { ChatMessage } from "@/features/chat/chat-message";

/** The next state of the answer turn, or null to remove it. */
export type TurnUpdate = (current: ChatMessage) => ChatMessage | null;

/** What a run says on its way through: the answer so far, and how it ended. */
const STOPPED_NOTE = "_Stopped before this answer was saved._";

/**
 * One question asked of one Document, and every way it can end.
 *
 * A rejected request is never rendered as an empty answer: each failure is
 * classified so a dead search, missing App Settings, and a broken connection
 * each read differently and offer the right recovery.
 *
 * An answer the reader walked away from is marked as not saved rather than left
 * looking stored, because the server recorded the turn as cancelled and the
 * transcript is not what will replay. An answer that never said anything is
 * removed instead, so a stop leaves no empty bubble behind.
 */
export async function runChatTurn({
  body,
  filename,
  botId,
  signal,
  isStale,
  patch,
}: {
  body: string;
  filename: string;
  botId: string;
  signal: AbortSignal;
  /** True once this turn belongs to a Document that is no longer open. */
  isStale: () => boolean;
  patch: (id: string, update: TurnUpdate) => void;
}): Promise<void> {
  try {
    await chatStream(
      body,
      filename,
      {
        onToken: (token) => {
          if (isStale()) return;
          patch(botId, (turn) => ({ ...turn, text: turn.text + token }));
        },
        onProviderError: (message, category) => {
          if (isStale()) return;
          patch(botId, (turn) => ({
            ...turn,
            text: message,
            failed: true,
            failure: category,
            needsSettings: isSettingsFailureMessage(message),
          }));
        },
        onPersistenceError: (message) => {
          if (isStale()) return;
          patch(botId, (turn) => ({ ...turn, text: message, failed: true, failure: "persistence" }));
        },
        onCancelled: (reason) => {
          if (isStale()) return;
          // The answer the user already read is not thrown away because it was
          // never stored. It is marked as not saved, the same as a stop.
          patch(botId, (turn) => ({
            ...turn,
            cancelled: true,
            stopReason: reason,
            text: stopped(turn),
          }));
        },
        onAbstained: ({ message, reason, retrieval }) => {
          if (isStale()) return;
          // Not a failure and not an empty answer: the app knows there is
          // nothing to ground one in, so it says so and cites nothing.
          patch(botId, (turn) => ({
            ...turn,
            text: message,
            abstained: true,
            abstentionReason: reason,
            retrieval,
          }));
        },
        onDone: ({ sources, claims, grounded, retrieval, truncated }) => {
          if (isStale()) return;
          patch(botId, (turn) => ({ ...turn, sources, claims, grounded, retrieval, truncated }));
        },
      },
      { signal },
    );
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") {
      patch(botId, (turn) => (turn.text ? { ...turn, cancelled: true, text: stopped(turn) } : null));
      return;
    }
    if (isStale()) return;
    const { failure, message, needsSettings } = classifyChatRequestError(e);
    patch(botId, (turn) => ({ ...turn, text: message, failed: true, failure, needsSettings }));
  }
}

/** The answer so far, plus the note that it was never stored. */
function stopped(turn: ChatMessage): string {
  return turn.text ? `${turn.text}\n\n${STOPPED_NOTE}` : "";
}