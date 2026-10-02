import { displayTitle } from "@/types/db";
import type { File as DbFile } from "@/types/db";
import type { IBriefDocument } from "@/services/research";
import { cn } from "@/lib/utils";

type Props = {
  files: DbFile[];
  selected: string[];
  onToggle: (file: DbFile) => void;
  /** The labels and titles the server assigned, once it has confirmed the pair. */
  scope: IBriefDocument[] | null;
};

/**
 * The Documents a brief may read, chosen from what is actually readable.
 *
 * A Document that is indexing, being removed, or carrying a stale index is not
 * offered at all rather than offered and refused: it cannot be part of a
 * scope, and a checkbox that cannot be checked is a worse affordance than one
 * that is not there. The labels A and B are the ones the brief itself uses,
 * shown on the selection rather than only in the result.
 */
export default function BriefScopePicker({ files, selected, onToggle, scope }: Props) {
  return (
    <fieldset>
      <legend className="label-meta">Scope · select exactly two</legend>
      {scope && (
        <p
          className="mt-1 font-mono text-[0.6rem] text-ink-faint"
          data-testid="brief-scope-labels"
        >
          {[...scope]
            .sort((a, b) => a.label.localeCompare(b.label))
            .map((item) => `[${item.label}] ${item.title}`)
            .join("  ·  ")}
        </p>
      )}
      <ul className="mt-2 max-h-56 space-y-px overflow-y-auto" data-testid="brief-scope-list">
        {files.map((file) => {
          const active = selected.includes(file.name);
          return (
            <li key={file.id}>
              <label
                className={cn(
                  "flex min-h-[44px] cursor-pointer items-center gap-2 border-l-2 px-3 py-2 text-xs",
                  active ? "border-marker bg-paper text-ink" : "border-rule text-ink-soft",
                )}
              >
                <input
                  type="checkbox"
                  checked={active}
                  onChange={() => onToggle(file)}
                  data-testid={`brief-scope-${file.name}`}
                  aria-label={`Include ${displayTitle(file)} in the brief`}
                  className="size-3.5 accent-[var(--color-ink)]"
                />
                <span className="min-w-0 flex-1 truncate">{displayTitle(file)}</span>
                {active && (
                  <span className="font-mono text-[0.6rem] text-marker">
                    {selected.indexOf(file.name) === 0 ? "A" : "B"}
                  </span>
                )}
              </label>
            </li>
          );
        })}
      </ul>
    </fieldset>
  );
}