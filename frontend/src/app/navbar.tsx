import { BookOpen } from "lucide-react";

export default function Navbar() {
  return (
    <header className="sticky top-0 z-50 w-full pt-4 pb-2">
      <div className="flex items-center justify-between glass rounded-xl px-6 py-4 shadow-sm">
        <div className="flex items-center gap-3 group cursor-pointer">
          <div className="bg-primary p-1.5 rounded text-white transition-transform group-hover:scale-105">
            <BookOpen size={18} strokeWidth={2} />
          </div>
          <h1 className="text-lg font-semibold tracking-tight text-slate-900 flex items-center">
            <span className="font-serif italic text-xl tracking-wide">PaperMind</span>
            <span className="text-[10px] font-sans font-medium text-slate-500 uppercase tracking-wider ml-2.5 border-l border-slate-200 pl-2.5">
              Research Workspace
            </span>
          </h1>
        </div>

        <div className="flex items-center gap-4">
          <div className="hidden md:flex items-center px-2.5 py-1 bg-slate-50 rounded text-xs font-semibold text-slate-600 border border-slate-200/80">
            Analysis Node v0.1.0
          </div>
        </div>
      </div>
    </header>
  );
}

