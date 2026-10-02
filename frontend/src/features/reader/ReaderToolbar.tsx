import {
  ChevronLeft,
  ChevronRight,
  List,
  MessageSquare,
  Minus,
  MoreHorizontal,
  PanelLeft,
  Plus,
  Trash2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu";
import { displayTitle } from "@/types/db";
import type { File as DbFile } from "@/types/db";

type Props = {
  file: DbFile | null;
  statusLine: string;
  zoom: number;
  onZoom: (next: number) => void;
  page: number;
  numPages: number | null;
  onPage: (page: number) => void;
  showStrip: boolean;
  onToggleStrip: () => void;
  onDelete: () => void;
  onOpenLibrary: () => void;
  onOpenChat: () => void;
};

/**
 * The reader's toolbar: which Document is open, how far into it the reader is,
 * and the three things they can do to the view.
 *
 * Zoom lives in the actions menu below the small breakpoint. With a 44px touch
 * target floor, eight controls need more width than a phone has, and the
 * overflowing group covered the library and chat buttons and swallowed their
 * clicks.
 */
export default function ReaderToolbar({
  file,
  statusLine,
  zoom,
  onZoom,
  page,
  numPages,
  onPage,
  showStrip,
  onToggleStrip,
  onDelete,
  onOpenLibrary,
  onOpenChat,
}: Props) {
  return (
    <header className="flex h-14 items-center justify-between gap-2 sm:gap-4 border-b border-rule bg-background/80 px-3 sm:px-5 backdrop-blur">
      <div className="flex min-w-0 items-center gap-3">
        <button
          type="button"
          onClick={onOpenLibrary}
          className="flex size-7 items-center justify-center rounded-sm border border-rule text-ink-soft hover:border-ink hover:text-ink lg:hidden"
          aria-label="Open library"
        >
          <PanelLeft className="size-3.5" />
        </button>
        <div className="min-w-0">
          <p className="truncate font-serif text-[0.95rem] leading-tight">
            {file ? displayTitle(file) : "Document Viewer"}
          </p>
          <p className="label-meta truncate">{statusLine}</p>
        </div>
      </div>

      <div className="flex items-center gap-2 sm:gap-4">
        {file && (
          <DropdownMenu>
            <DropdownMenuTrigger
              render={
                <button
                  type="button"
                  className="flex size-7 items-center justify-center rounded-sm border border-rule text-ink-soft hover:border-ink hover:text-ink"
                  aria-label="Document actions"
                  title="Document actions"
                />
              }
            >
              <MoreHorizontal className="size-3.5" />
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="border-rule bg-paper">
              <DropdownMenuItem className="sm:hidden" onClick={() => onZoom(zoom - 10)}>
                <Minus className="size-3.5" /> Zoom out
              </DropdownMenuItem>
              <DropdownMenuItem className="sm:hidden" onClick={() => onZoom(zoom + 10)}>
                <Plus className="size-3.5" /> Zoom in
              </DropdownMenuItem>
              <DropdownMenuItem
                className="text-destructive focus:bg-destructive/10 focus:text-destructive"
                onClick={onDelete}
              >
                <Trash2 className="size-3.5" /> Delete Permanently
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        )}
        {file && (
          <button
            type="button"
            onClick={onToggleStrip}
            className={cn(
              "flex size-7 items-center justify-center rounded-sm border transition-colors",
              showStrip
                ? "border-ink bg-ink text-paper"
                : "border-rule text-ink-soft hover:border-ink",
            )}
            aria-pressed={showStrip}
            aria-label={showStrip ? "Hide page strip" : "Show page strip"}
            title={showStrip ? "Hide page strip" : "Show page strip"}
          >
            <List className="size-3.5" />
          </button>
        )}

        <div className="hidden items-center gap-1 border-l border-rule pl-2 sm:flex sm:pl-4">
          <button
            type="button"
            onClick={() => onZoom(zoom - 10)}
            className="flex size-6 items-center justify-center text-ink-soft hover:text-ink"
            aria-label="Zoom out"
          >
            <Minus className="size-3" />
          </button>
          <span className="w-10 text-center font-mono text-[0.68rem] text-ink-soft">{zoom}%</span>
          <button
            type="button"
            onClick={() => onZoom(zoom + 10)}
            className="flex size-6 items-center justify-center text-ink-soft hover:text-ink"
            aria-label="Zoom in"
          >
            <Plus className="size-3" />
          </button>
        </div>

        <div className="flex items-center gap-1 border-l border-rule pl-2 sm:pl-4">
          <button
            type="button"
            onClick={() => onPage(page - 1)}
            className="flex size-6 items-center justify-center text-ink-soft hover:text-ink"
            aria-label="Previous page"
            data-testid="page-prev"
          >
            <ChevronLeft className="size-3.5" />
          </button>
          <span className="font-mono text-[0.68rem] text-ink-soft" data-testid="page-indicator">
            {String(page).padStart(2, "0")} / {String(numPages ?? 0).padStart(2, "0")}
          </span>
          <button
            type="button"
            onClick={() => onPage(page + 1)}
            className="flex size-6 items-center justify-center text-ink-soft hover:text-ink"
            aria-label="Next page"
            data-testid="page-next"
          >
            <ChevronRight className="size-3.5" />
          </button>
        </div>
        <button
          type="button"
          onClick={onOpenChat}
          className="flex size-7 items-center justify-center rounded-sm border border-rule text-ink-soft hover:border-ink hover:text-ink lg:hidden"
          aria-label="Open chat"
        >
          <MessageSquare className="size-3.5" />
        </button>
      </div>
    </header>
  );
}
