import { useState } from "react";
import { useOpenEvidence } from "@/hooks/useOpenEvidence";
import { cn } from "@/lib/utils";
import type { IBriefEvidence } from "@/services/research";

/**
 * The Passages a brief collected, each openable at the Page it came from.
 *
 * This is the part of a brief a reader can check. Everything the model wrote is
 * a claim; every claim here is a passage the app actually retrieved, which is
 * what makes the two separable.
 */
export default function BriefEvidenceList({ evidence }: { evidence: IBriefEvidence[] }) {
  const openEvidence = useOpenEvidence();
  const [open, setOpen] = useState(true);
  if (evidence.length === 0) return null;

  return (
    <div className="mt-4 border-t border-rule pt-3">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        aria-label={`${open ? "Hide" : "Show"} ${evidence.length} passages found`}
        className="flex w-full items-center justify-between text-ink-faint hover:text-ink"
      >
        <span className="label-meta">{evidence.length} passages found</span>
      </button>
      {open && (
        <ol className="mt-2 space-y-2" data-testid="brief-evidence">
          {evidence.map((item) => (
            <li key={item.evidence_id} className="border-l-2 border-rule pl-3">
              <button
                type="button"
                disabled={item.page == null}
                onClick={() => openEvidence(item)}
                data-testid={`brief-evidence-${item.evidence_id}`}
                data-document={item.document_id}
                data-page={item.page ?? ""}
                className={cn(
                  "block w-full text-left",
                  item.page != null ? "cursor-pointer hover:border-marker" : "cursor-default",
                )}
                title={item.page != null ? `Open page ${item.page}` : "This passage has no page"}
              >
                <span className="font-mono text-[0.6rem] text-ink-faint">
                  <span className="text-marker">[{item.evidence_id}]</span> {item.title}
                  {item.page != null ? ` · p. ${item.page}` : ""} · rank {item.rank}
                  {item.read ? "" : " · excerpt"}
                </span>
                <span className="mt-0.5 block font-serif text-xs italic leading-snug text-ink-soft">
                  “{item.content.slice(0, 200)}”
                </span>
              </button>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}