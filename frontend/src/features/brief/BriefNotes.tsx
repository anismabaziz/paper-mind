import type { IBriefBudget, IBriefModel } from "@/services/research";

/** The questions the two Documents could not settle. */
export function BriefGapList({ gaps }: { gaps: string[] | undefined }) {
  const list = gaps ?? [];
  if (list.length === 0) return null;
  return (
    <div className="mt-4" data-testid="brief-gaps">
      <p className="label-meta">Unresolved gaps</p>
      <ul className="mt-1 list-disc space-y-1 pl-5">
        {list.map((gap, index) => (
          <li key={index} className="font-serif text-xs leading-relaxed text-ink-soft">
            {gap}
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Prompt version and model, so a saved brief stays comparable. */
export function BriefMeta({
  promptVersion,
  model,
}: {
  promptVersion: string | null | undefined;
  model: IBriefModel | null | undefined;
}) {
  if (!promptVersion && !model) return null;
  return (
    <p className="mt-3 font-mono text-[0.6rem] text-ink-faint" data-testid="brief-meta">
      {promptVersion && <span>Prompt {promptVersion}</span>}
      {promptVersion && model && <span> · </span>}
      {model && <span>{model.provider}/{model.model}</span>}
    </p>
  );
}

/**
 * What the loop spent, and against which ceiling.
 *
 * Shown because a partial brief is only interpretable next to its limits: six
 * of six turns is a different situation from six turns of eight, and a reader
 * who cannot see the difference cannot tell which one they are looking at.
 */
export function BriefBudgetLine({ budget }: { budget: IBriefBudget }) {
  if (!budget.max_turns) return null;
  return (
    <span className="font-mono text-[0.6rem] text-ink-faint" data-testid="brief-budget">
      {budget.turns}/{budget.max_turns} turns · {budget.tool_calls}/{budget.max_tool_calls} reads
    </span>
  );
}