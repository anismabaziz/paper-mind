import type {
  IBrief,
  IBriefBudget,
  IBriefClaim,
  IBriefDocument,
  IBriefEvidence,
  IBriefModel,
} from "@/services/research";

/** What a run ended as, in the one word the reader reads for it. */
export const OUTCOME_LABEL: Record<string, string> = {
  complete: "complete",
  partial: "partial",
  incomplete: "incomplete",
  cancelled: "cancelled",
  failed: "failed",
  abstained: "abstained",
};

/**
 * A brief that is running, or the result it produced.
 *
 * A stopped run is a result, not an absence of one: it carries the evidence
 * collected before it stopped and an incomplete status, because that is
 * exactly what the reader is left with.
 */
export type Run =
  | { state: "idle" }
  | { state: "running"; documents: IBriefDocument[]; briefId: string | null }
  | {
      state: "done";
      status: "complete" | "incomplete";
      answer: string;
      stoppedBy: string | null;
      message: string | null;
      evidence: IBriefEvidence[];
      budget: IBriefBudget;
      documents: IBriefDocument[];
      brief: IBrief | null;
      claims: IBriefClaim[];
      gaps: string[];
      abstained: boolean;
      promptVersion: string | null;
      model: IBriefModel | null;
    }
  | {
      state: "abstained";
      message: string;
      documents: IBriefDocument[];
      stoppedBy: string | null;
    }
  | {
      state: "failed";
      message: string;
      evidence: IBriefEvidence[];
      documents: IBriefDocument[];
    };

/** A budget for a run that was stopped before it could report one. */
export const NO_BUDGET: IBriefBudget = {
  turns: 0,
  tool_calls: 0,
  repeated_calls: 0,
  tokens: 0,
  elapsed_seconds: 0,
  documents: 2,
  max_turns: 0,
  max_tool_calls: 0,
  max_repeated_calls: 0,
  max_tokens: 0,
  max_seconds: 0,
};

/**
 * The outcome word for a finished run. Cancelled, partial and abstained are
 * three different things that a single "incomplete" would flatten.
 */
export function outcomeOf(
  status: "complete" | "incomplete",
  stoppedBy: string | null,
  abstained: boolean,
): string {
  if (abstained) return OUTCOME_LABEL.abstained;
  if (status === "complete") return OUTCOME_LABEL.complete;
  if (stoppedBy === "cancelled") return OUTCOME_LABEL.cancelled;
  return OUTCOME_LABEL.partial;
}

/** Turn a stop reason into the word a reader reads. */
export function humanize(reason: string): string {
  return reason.replace(/_/g, " ");
}