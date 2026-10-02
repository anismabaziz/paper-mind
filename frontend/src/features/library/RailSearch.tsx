import { Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

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
      <Label
        htmlFor="library-search"
        className="flex h-9 items-center gap-2 border border-rule bg-paper px-3 font-normal focus-within:border-ink"
      >
        <Search className="size-3.5 shrink-0 text-ink-faint" aria-hidden="true" />
        <Input
          id="library-search"
          value={query}
          onChange={(e) => onQuery(e.target.value)}
          placeholder="Search library"
          aria-label="Search library"
          data-testid="library-search"
          className="h-auto min-w-0 flex-1 rounded-none border-0 bg-transparent px-0 py-0 text-xs shadow-none placeholder:text-ink-faint focus-visible:ring-0"
        />
      </Label>
    </div>
  );
}