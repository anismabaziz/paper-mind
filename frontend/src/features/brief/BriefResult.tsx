import { Loader2 } from "lucide-react";
import { MarkdownRenderer } from "@/components/MarkdownRenderer";
import type { Run } from "./brief-run";
import { OUTCOME_LABEL, humanize, outcomeOf } from "./brief-run";
import BriefClaimList from "./BriefClaimList";
import BriefEvidenceList from "./BriefEvidenceList";
import { BriefBudgetLine, BriefGapList, BriefMeta } from "./BriefNotes";

/**
 * What one run produced, labelled by how it ended.
 *
 * The four endings are four different things and none of them is a failure by
 * default: a brief that finished, one that ran out of budget, one the model
 * declined to write because the Documents settled nothing, and one the server
 * refused. Only the last is shown in red.
 */
export default function BriefResult({ run }: { run: Run }) {
  if (run.state === "idle") return null;

  if (run.state === "running") {
    return (
      <p
        role="status"
        className="mt-5 font-mono text-xs text-ink-faint"
        data-testid="brief-running"
      >
        <Loader2 size={12} className="mr-2 inline animate-spin" />
        Searching the two Documents…
      </p>
    );
  }

  if (run.state === "failed") {
    return (
      <div
        role="alert"
        className="mt-5 rounded-sm border border-destructive/30 bg-destructive/5 p-3"
        data-testid="brief-failed"
        data-outcome="failed"
      >
        <p className="font-mono text-xs font-semibold text-destructive">{OUTCOME_LABEL.failed}</p>
        <p className="mt-1 font-mono text-[0.65rem] leading-relaxed text-ink-soft">{run.message}</p>
        <BriefEvidenceList evidence={run.evidence} />
      </div>
    );
  }

  if (run.state === "abstained") {
    return (
      <div
        role="status"
        className="mt-5 rounded-sm border border-rule bg-canvas p-3"
        data-testid="brief-abstained"
        data-outcome="abstained"
      >
        <p className="label-meta">{OUTCOME_LABEL.abstained}</p>
        <p className="mt-1 font-serif text-xs leading-relaxed text-ink-soft">{run.message}</p>
        {run.stoppedBy && (
          <p className="mt-1 font-mono text-[0.6rem] text-ink-faint">
            Stopped by {humanize(run.stoppedBy)}.
          </p>
        )}
      </div>
    );
  }

  const outcome = outcomeOf(run.status, run.stoppedBy, run.abstained ?? false);

  return (
    <div
      className="mt-5 border-t border-rule pt-4"
      data-testid="brief-result"
      data-status={run.status}
      data-outcome={outcome}
    >
      <div className="flex items-center justify-between gap-2">
        <p className="label-meta" data-testid="brief-outcome">
          Brief · {outcome}
          {run.stoppedBy ? ` · ${humanize(run.stoppedBy)}` : ""}
        </p>
        <BriefBudgetLine budget={run.budget} />
      </div>
      {run.message && (
        <p
          className="mt-2 font-mono text-[0.65rem] leading-relaxed text-ink-soft"
          data-testid="brief-incomplete-note"
        >
          {run.message}
        </p>
      )}
      {run.answer && <MarkdownRenderer text={run.answer} />}
      <BriefClaimList claims={run.claims} evidence={run.evidence} />
      <BriefGapList gaps={run.gaps} />
      <BriefEvidenceList evidence={run.evidence} />
      <BriefMeta promptVersion={run.promptVersion} model={run.model} />
    </div>
  );
}