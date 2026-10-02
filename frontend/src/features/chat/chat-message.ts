import type {
  ChatAbstentionReason,
  ChatOutcomeFailure,
  IClaim,
  IRetrievalResult,
  ISource,
} from "@/services/files";

/**
 * One turn in the conversation, as the pane renders it.
 *
 * A turn is more than its text: how it ended decides whether its citations are
 * shown at all, so the outcome travels with the message rather than being
 * worked out at the point of display.
 */
export type ChatMessage = {
  id: string;
  text: string;
  sender: "user" | "bot";
  sources?: ISource[];
  claims?: IClaim[];
  /** At least one claim named a passage the app actually supplied. */
  grounded?: boolean;
  retrieval?: IRetrievalResult;
  failed?: boolean;
  needsSettings?: boolean;
  /** The answer was cut short by the model's output budget. */
  truncated?: boolean;
  /** True once the user walked away from this answer. */
  cancelled?: boolean;
  /** Why the answer was stopped, when the server said. */
  stopReason?: string;
  /** Which outcome ended this answer, for styling and for tests. */
  failure?: ChatOutcomeFailure;
  /** The app answered without a model call: there was no evidence. */
  abstained?: boolean;
  /** Why it abstained, which decides what the reader can do next. */
  abstentionReason?: ChatAbstentionReason;
  /** The question this answer was started for, so a failure can offer a retry. */
  question?: string;
};

/** How an answer ended, as the machine-readable outcome and the word for it. */
export type Outcome = { outcome: string | undefined; label: string };

const GROUNDED: Outcome = { outcome: undefined, label: "grounded" };

/**
 * Retrieval scores are a ranking, not a chance of being right, so only a claim
 * that cites earns this.
 */
const UNGROUNDED: Outcome = { outcome: "ungrounded", label: "ungrounded" };

/** Nothing has been decided yet: no claim has been read, so nothing is claimed. */
const ANSWERING: Outcome = { outcome: "answering", label: "answering" };

export function describe(m: ChatMessage): Outcome {
  if (m.failure) return { outcome: m.failure, label: "failed" };
  if (m.cancelled) return { outcome: "cancelled", label: "stopped" };
  if (m.abstained) return { outcome: m.abstentionReason, label: "abstained" };
  if (m.truncated) return { outcome: "truncated", label: "truncated" };
  // The terminal event is what settles whether anything is cited, so an answer
  // still arriving is not called grounded before it is.
  if (m.sources === undefined) return ANSWERING;
  if (m.grounded === false) return UNGROUNDED;
  return GROUNDED;
}

/**
 * What the reader can do about an abstention.
 *
 * Nothing to read is not a fault: the paper simply does not cover the question.
 * Matches that could not be used are a different case, and a reindex is the
 * only thing that clears them.
 */
export const ABSTENTION_HINTS: Record<ChatAbstentionReason, string> = {
  no_evidence: "Nothing in this paper answers that. Try a different question, or one of the prompts above.",
  evidence_unusable: "The passages this question matched could not be read. Reindex the paper to search it again.",
};

/**
 * What each failure means and what the reader can do next.
 *
 * Every failed answer keeps this shape: a heading the server did not write,
 * the server's safe message, one recovery action, and no citation list. A
 * failed answer never renders as grounded, however much text arrived first.
 */
export const FAILURE_COPY: Record<
  ChatOutcomeFailure,
  { heading: string; hint: string; retryLabel: string }
> = {
  provider: {
    heading: "The model could not answer",
    hint: "The paper was searched but the model call failed. Your question is kept — try it again.",
    retryLabel: "Ask again",
  },
  timeout: {
    heading: "The answer took too long",
    hint: "The model call was stopped before it finished. A narrower question usually completes faster.",
    retryLabel: "Try again",
  },
  empty_output: {
    heading: "The model returned nothing",
    hint: "The model call finished with no text. Rephrase the question and try again.",
    retryLabel: "Try again",
  },
  citations: {
    heading: "The answer cited evidence it was not given",
    hint: "The model named a passage outside what was retrieved, so the answer was withheld rather than shown ungrounded. Ask again — nothing here is cited.",
    retryLabel: "Ask again",
  },
  retrieval_unavailable: {
    heading: "Search is unavailable",
    hint: "The paper could not be searched right now, so no model was asked. Wait a moment, then try the question again.",
    retryLabel: "Retry search",
  },
  persistence: {
    heading: "The answer could not be saved",
    hint: "The answer arrived but storing it failed, so this Conversation does not include it. Ask the question again to store a fresh answer.",
    retryLabel: "Ask again",
  },
  interrupted: {
    heading: "The answer was interrupted",
    hint: "The connection broke before the answer finished. What arrived is not saved as a complete answer — try the question again.",
    retryLabel: "Try again",
  },
  settings: {
    heading: "No chat provider is configured",
    hint: "The question was never sent to a model. Verify a provider and API key in App Settings, then ask the question again.",
    retryLabel: "Ask again",
  },
};

export const suggestedPrompts = [
  "What is the main topic?",
  "Summarize key findings",
  "Methodology used?",
];

export const METHOD_LABELS: Record<IRetrievalResult["method"], string> = {
  dense: "dense retrieval",
  sparse: "sparse retrieval",
  hybrid: "hybrid retrieval",
};

export function formatManifestValue(value: string | number | boolean | null): string {
  if (typeof value === "boolean") return value ? "on" : "off";
  return value === null || value === "" ? "unpinned" : String(value);
}