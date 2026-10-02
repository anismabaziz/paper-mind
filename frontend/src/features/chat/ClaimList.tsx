import usePdfStore from "@/store/pdf-state";
import { cn } from "@/lib/utils";
import type { IClaim, ISource } from "@/services/files";

/**
 * What the answer claimed, and what it cited for each claim.
 *
 * A claim with no citation is shown as unsupported rather than dropped. Leaving
 * it out would make an answer look better supported than it is.
 */
export default function ClaimList({ claims, sources }: { claims: IClaim[]; sources: ISource[] }) {
  const byId = new Map(sources.map((source) => [source.source_id, source]));

  return (
    <div className="mt-4 border-t border-rule pt-3">
      <p className="label-meta">Claims this answer makes</p>
      <ol className="mt-2.5 space-y-1.5">
        {claims.map((claim, index) => (
          <li key={index} className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
            <span className="font-serif text-[0.85rem] leading-snug text-ink-soft">{claim.claim}</span>
            {claim.sources.length > 0 ? (
              <span className="flex items-baseline gap-1">
                {claim.sources.map((sourceId) => {
                  const source = byId.get(sourceId);
                  return source ? <ClaimCitation key={sourceId} source={source} /> : null;
                })}
              </span>
            ) : (
              <span className="font-mono text-[0.6rem] text-ink-faint">not supported</span>
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}

/** A claim's citation, which opens the page that supports it when it has one. */
function ClaimCitation({ source }: { source: ISource }) {
  const setCitationTarget = usePdfStore((s) => s.setCitationTarget);
  const hasPage = source.page != null;

  return (
    <button
      type="button"
      disabled={!hasPage}
      onClick={() => hasPage && setCitationTarget(source.page as number)}
      title={hasPage ? `Open page ${source.page}` : "This passage has no page to open"}
      aria-label={
        hasPage
          ? `Open page ${source.page} for citation ${source.source_id}`
          : `Citation ${source.source_id}, no page to open`
      }
      data-testid={`citation-${source.source_id}`}
      className={cn(
        "font-mono text-[0.62rem] tracking-wide",
        hasPage ? "cursor-pointer text-marker hover:underline" : "text-ink-faint",
      )}
    >
      {source.source_id}
    </button>
  );
}