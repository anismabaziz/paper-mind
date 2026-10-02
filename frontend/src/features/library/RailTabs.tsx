import { Archive, Clock3 } from "lucide-react";
import { cn } from "@/lib/utils";

function pad(n: number) {
  return String(n).padStart(2, "0");
}

type Tab = "library" | "recent";

type Props = {
  tab: Tab;
  onTab: (tab: Tab) => void;
  totalCount: number;
  recentCount: number;
  listTitle: string;
};

/**
 * The rail's two views of the same library, and the heading over whichever one
 * is open. Recent readings is a filter over the same Documents rather than a
 * separate collection, so both tabs list the same rows.
 */
export default function RailTabs({
  tab,
  onTab,
  totalCount,
  recentCount,
  listTitle,
}: Props) {
  return (
    <>
      <p className="label-meta px-2 pb-2">Workspace</p>
      <ul className="mb-6 space-y-0.5">
          <RailTab
            icon={<Archive className="size-3.5" />}
            label="Library"
            count={totalCount}
            active={tab === "library"}
            onSelect={() => onTab("library")}
          />
          <RailTab
            icon={<Clock3 className="size-3.5" />}
            label="Recent readings"
            count={recentCount}
            active={tab === "recent"}
            onSelect={() => onTab("recent")}
          />
      </ul>

      <div className="flex items-center justify-between px-2 pb-2">
        <p className="label-meta">{listTitle}</p>
        {tab !== "library" && (
          <button
            type="button"
            onClick={() => onTab("library")}
            className="font-mono text-[0.6rem] text-ink-faint underline-offset-2 hover:text-marker hover:underline"
          >
            Show all
          </button>
        )}
      </div>
    </>
  );
}

function RailTab({
  icon,
  label,
  count,
  active,
  onSelect,
}: {
  icon: React.ReactNode;
  label: string;
  count: number;
  active: boolean;
  onSelect: () => void;
}) {
  return (
    <li>
      <button
        type="button"
        onClick={onSelect}
        aria-pressed={active}
        className={cn(
          "flex w-full items-center gap-2.5 px-2 py-2 text-left text-xs",
          active ? "bg-marker-soft font-medium text-marker" : "text-ink-soft hover:bg-canvas",
        )}
      >
        {icon}
        <span className="flex-1">{label}</span>
        <span className="font-mono text-[0.6rem] text-ink-faint">{pad(count)}</span>
      </button>
    </li>
  );
}