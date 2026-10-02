import { Search } from "lucide-react";

/**
 * The rail's search. It matches on the title a reader recognises rather than
 * on the storage filename, because the filename is never what they saw.
 */
export default function RailSearch({
  query,
  onQuery,
}: {
  query: string;
  onQuery: (value: string) => void;
}) {
  return (
    <div className="border-b border-rule p-4">
      <label className="flex h-9 items-center gap-2 border border-rule bg-paper px-3 focus-within:border-ink">
        <Search className="size-3.5 text-ink-faint" />
        <input
          value={query}
          onChange={(e) => onQuery(e.target.value)}
          placeholder="Search library"
          aria-label="Search library"
          className="min-w-0 flex-1 bg-transparent text-xs outline-none placeholder:text-ink-faint"
          data-testid="library-search"
        />
      </label>
    </div>
  );
}