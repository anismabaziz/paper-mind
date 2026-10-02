import { useOpenEvidence } from "@/hooks/useOpenEvidence";
import type { IBriefClaim, IBriefEvidence } from "@/services/research";

/**
 * The ordered claims, with supporting and conflicting evidence kept apart.
 *
 * A claim with nothing behind it is listed as unresolved rather than dropped.
 * Leaving it out would make the brief look more settled than the evidence
 * supports, which is the one thing a reader is relying on it not to do.
 */
export default function BriefClaimList({
  claims,
  evidence,
}: {
  claims: IBriefClaim[] | undefined;
  evidence: IBriefEvidence[];
}) {
  const openEvidence = useOpenEvidence();
  const list = claims ?? [];
  if (list.length === 0) return null;
  const byId = new Map(evidence.map((item) => [item.evidence_id, item]));

  const cite = (id: string, prefix: string, order: number) => {
    const item = byId.get(id);
    const hasPage = item?.page != null;
    return (
      <button
        key={`${prefix}-${id}`}
        type="button"
        disabled={!hasPage}
        onClick={() => openEvidence(item)}
        data-testid={`brief-claim-${order}-${prefix}-${id}`}
        title={hasPage ? `Open ${item?.title ?? ""} page ${item?.page}` : "This passage has no page"}
        className={hasPage ? "underline underline-offset-2 hover:text-ink" : ""}
      >
        [{id}]
      </button>
    );
  };

  return (
    <ol className="mt-4 space-y-2" data-testid="brief-claims">
      {[...list]
        .sort((a, b) => a.order - b.order)
        .map((claim) => (
          <li
            key={claim.order}
            className="border-l-2 border-marker/60 pl-3"
            data-testid={`brief-claim-${claim.order}`}
            data-status={claim.status}
          >
            <p className="font-serif text-xs leading-relaxed text-ink">{claim.claim}</p>
            <p className="mt-1 font-mono text-[0.6rem] text-ink-faint">
              <span data-testid={`brief-claim-${claim.order}-status`}>{claim.status}</span>
              {claim.supports.length > 0 && (
                <span>
                  {" "}
                  · supports{" "}
                  {claim.supports.map((id) => cite(id, "supports", claim.order))}
                </span>
              )}
              {claim.conflicts.length > 0 && (
                <span>
                  {" "}
                  · conflicts{" "}
                  {claim.conflicts.map((id) => cite(id, "conflicts", claim.order))}
                </span>
              )}
              {claim.supports.length === 0 && claim.conflicts.length === 0 && (
                <span> · unresolved — no evidence</span>
              )}
            </p>
          </li>
        ))}
    </ol>
  );
}