function FakePageBars() {
  return (
    <span className="flex h-full flex-col gap-[3px]">
      {Array.from({ length: 11 }).map((_, i) => (
        <span key={i} className="block h-[2px] rounded-full bg-ink/15" style={{ width: `${55 + ((i * 37) % 45)}%` }} />
      ))}
    </span>
  );
}

export function ThumbnailPlaceholder({ count }: { count: number }) {
  return (
    <div className="flex gap-2 overflow-hidden">
      {Array.from({ length: count }, (_, i) => i + 1).map((n) => (
        <div key={n} className="relative h-28 aspect-[3/4] shrink-0 overflow-hidden rounded-[2px] border border-rule bg-paper p-1.5 opacity-40 box-border max-w-full min-w-0">
          <FakePageBars />
          <span className="absolute right-1 bottom-1 font-mono text-[0.55rem] text-ink-faint">{n}</span>
        </div>
      ))}
    </div>
  );
}
